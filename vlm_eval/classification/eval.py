import argparse
import os

from einops import repeat
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from open_flamingo.eval.classification_utils import (
    IMAGENET_CLASSNAMES,
    IMAGENET_1K_CLASS_ID_TO_LABEL,
    HM_CLASSNAMES,
    HM_CLASS_ID_TO_LABEL
)
from open_flamingo.eval.eval_datasets import (
    HatefulMemesDataset, )
from open_flamingo.eval.eval_datasets import ImageNetDataset
from open_flamingo.eval.eval_model import BaseEvalModel
from vlm_eval.utils import *


def evaluate_classification(
        args: argparse.Namespace,
        eval_model,
        seed: int = 42,
        num_shots: int = 8,
        no_kv_caching=False,
        dataset_name: str = "imagenet",
):
    """
    Evaluate a model on classification dataset.

    Args:
        eval_model (BaseEvalModel): model to evaluate
        imagenet_root (str): path to imagenet root for the specified split.
        seed (int, optional): random seed. Defaults to 42.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (str, optional): dataset name. Defaults to "imagenet".

    Returns:
        float: accuracy score
    """
    if args.model != "open_flamingo":
        raise NotImplementedError(
            "evaluate_classification is currently only supported for OpenFlamingo "
            "models"
        )
    batch_size = args.batch_size
    num_samples = args.num_samples
    model, tokenizer = eval_model.model, eval_model.tokenizer

    if dataset_name == "imagenet":
        train_dataset = ImageNetDataset(os.path.join(args.imagenet_root, "../train"))
        test_dataset = ImageNetDataset(os.path.join(args.imagenet_root, "val"))
    elif dataset_name == "hateful_memes":
        train_dataset = HatefulMemesDataset(
            args.hateful_memes_image_dir_path,
            args.hateful_memes_train_annotations_json_path,
        )
        test_dataset = HatefulMemesDataset(
            args.hateful_memes_image_dir_path,
            args.hateful_memes_test_annotations_json_path,
        )
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        batch_size,
        seed,
    )

    acc1 = 0
    acc5 = 0

    if dataset_name == "imagenet":
        prompt_text = "<image>Output:"
    elif dataset_name == "hateful_memes":
        prompt_text = "<image>is an image with: '{meme_text}' written on it. Is it hateful? Answer: "

    predictions = []

    np.random.seed(seed)
    for batch_idx, batch in tqdm(
            enumerate(test_dataloader),
            desc=f"Running inference {dataset_name}",
    ):
        batch_images = []
        batch_text = []

        for idx in range(len(batch["image"])):
            # Choose a different set of random context samples for each sample
            # from the training set
            context_indices = np.random.choice(
                len(train_dataset), effective_num_shots, replace=False
            )

            in_context_samples = [train_dataset[i] for i in context_indices]

            if num_shots > 0:
                vision_x = [
                    eval_model.image_processor(data["image"]).unsqueeze(0)
                    for data in in_context_samples
                ]
            else:
                vision_x = []

            vision_x = vision_x + [
                eval_model.image_processor(batch["image"][idx]).unsqueeze(0)
            ]
            batch_images.append(torch.cat(vision_x, dim=0))

            def sample_to_prompt(sample):
                if dataset_name == "hateful_memes":
                    return prompt_text.replace("{meme_text}", sample["ocr"])
                else:
                    return prompt_text

            context_text = "".join(
                f"{sample_to_prompt(in_context_samples[i])}{in_context_samples[i]['class_name']}<|endofchunk|>"
                for i in range(effective_num_shots)
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(context_text)

        # shape [B, T_img, C, h, w]
        vision_x = torch.stack(batch_images, dim=0)
        # shape [B, T_img, 1, C, h, w] where 1 is the frame dimension
        vision_x = vision_x.unsqueeze(2)

        # Cache the context text: tokenize context and prompt,
        # e.g. '<context> a picture of a '
        text_x = [
            context_text + sample_to_prompt({k: batch[k][idx] for k in batch.keys()})
            for idx, context_text in enumerate(batch_text)
        ]

        ctx_and_prompt_tokenized = tokenizer(
            text_x,
            return_tensors="pt",
            padding="longest",
            max_length=2000,
        )

        ctx_and_prompt_input_ids = ctx_and_prompt_tokenized["input_ids"].to(
            eval_model.device
        )
        ctx_and_prompt_attention_mask = (
            ctx_and_prompt_tokenized["attention_mask"].to(eval_model.device).bool()
        )

        def _detach_pkvs(pkvs):
            """Detach a set of past key values."""
            return list([tuple([x.detach() for x in inner]) for inner in pkvs])

        if not no_kv_caching:
            eval_model.cache_media(
                input_ids=ctx_and_prompt_input_ids,
                vision_x=vision_x.to(eval_model.device),
            )

            with torch.no_grad():
                precomputed = eval_model.model(
                    vision_x=None,
                    lang_x=ctx_and_prompt_input_ids,
                    attention_mask=ctx_and_prompt_attention_mask,
                    clear_conditioned_layers=False,
                    use_cache=True,
                )

            precomputed_pkvs = _detach_pkvs(precomputed.past_key_values)
            precomputed_logits = precomputed.logits.detach()
        else:
            precomputed_pkvs = None
            precomputed_logits = None

        if dataset_name == "imagenet":
            all_class_names = IMAGENET_CLASSNAMES
        else:
            all_class_names = HM_CLASSNAMES

        if dataset_name == "imagenet":
            class_id_to_name = IMAGENET_1K_CLASS_ID_TO_LABEL
        else:
            class_id_to_name = HM_CLASS_ID_TO_LABEL

        overall_probs = []
        for class_name in all_class_names:
            past_key_values = None
            # Tokenize only the class name and iteratively decode the model's
            # predictions for this class.
            classname_tokens = tokenizer(
                class_name, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(eval_model.device)

            if classname_tokens.ndim == 1:  # Case: classname is only 1 token
                classname_tokens = torch.unsqueeze(classname_tokens, 1)

            classname_tokens = repeat(
                classname_tokens, "b s -> (repeat b) s", repeat=len(batch_text)
            )

            if not no_kv_caching:
                # Compute the outputs one token at a time, using cached
                # activations.

                # Initialize the elementwise predictions with the last set of
                # logits from precomputed; this will correspond to the predicted
                # probability of the first position/token in the imagenet
                # classname. We will append the logits for each token to this
                # list (each element has shape [B, 1, vocab_size]).
                elementwise_logits = [precomputed_logits[:, -2:-1, :]]

                for token_idx in range(classname_tokens.shape[1]):
                    _lang_x = classname_tokens[:, token_idx].reshape((-1, 1))
                    outputs = eval_model.get_logits(
                        lang_x=_lang_x,
                        past_key_values=(
                            past_key_values if token_idx > 0 else precomputed_pkvs
                        ),
                        clear_conditioned_layers=False,
                    )
                    past_key_values = _detach_pkvs(outputs.past_key_values)
                    elementwise_logits.append(outputs.logits.detach())

                # logits/probs has shape [B, classname_tokens + 1, vocab_size]
                logits = torch.concat(elementwise_logits, 1)
                probs = torch.softmax(logits, dim=-1)

                # collect the probability of the generated token -- probability
                # at index 0 corresponds to the token at index 1.
                probs = probs[:, :-1, :]  # shape [B, classname_tokens, vocab_size]

                gen_probs = (
                    torch.gather(probs, 2, classname_tokens[:, :, None])
                    .squeeze(-1)
                    .cpu()
                )

                class_prob = torch.prod(gen_probs, 1).numpy()
            else:
                # Compute the outputs without using cached
                # activations.

                # contatenate the class name tokens to the end of the context
                # tokens
                _lang_x = torch.cat([ctx_and_prompt_input_ids, classname_tokens], dim=1)
                _attention_mask = torch.cat(
                    [
                        ctx_and_prompt_attention_mask,
                        torch.ones_like(classname_tokens).bool(),
                    ],
                    dim=1,
                )

                outputs = eval_model.get_logits(
                    vision_x=vision_x.to(eval_model.device),
                    lang_x=_lang_x.to(eval_model.device),
                    attention_mask=_attention_mask.to(eval_model.device),
                    clear_conditioned_layers=True,
                )

                logits = outputs.logits.detach().float()
                probs = torch.softmax(logits, dim=-1)

                # get probability of the generated class name tokens
                gen_probs = probs[
                            :, ctx_and_prompt_input_ids.shape[1] - 1: _lang_x.shape[1], :
                            ]
                gen_probs = (
                    torch.gather(gen_probs, 2, classname_tokens[:, :, None])
                    .squeeze(-1)
                    .cpu()
                )
                class_prob = torch.prod(gen_probs, 1).numpy()

            overall_probs.append(class_prob)

        overall_probs = np.row_stack(overall_probs).T  # shape [B, num_classes]

        eval_model.uncache_media()

        def topk(probs_ary: np.ndarray, k: int) -> np.ndarray:
            """Return the indices of the top k elements in probs_ary."""
            return np.argsort(probs_ary)[::-1][:k]

        for i in range(len(batch_text)):
            highest_prob_idxs = topk(overall_probs[i], 5)

            top5 = [class_id_to_name[pred] for pred in highest_prob_idxs]

            y_i = batch["class_name"][i]
            acc5 += int(y_i in set(top5))
            acc1 += int(y_i == top5[0])

            predictions.append(
                {
                    "id": batch["id"][i],
                    "gt_label": y_i,
                    "pred_label": top5[0],
                    "pred_score": overall_probs[i][highest_prob_idxs[0]]
                    if dataset_name == "hateful_memes"
                    else None,  # only for hateful memes
                }
            )

    # all gather
    all_predictions = [None] * args.world_size
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists

    all_predictions = [
        item for sublist in all_predictions for item in sublist
    ]  # flatten

    # Hack to remove samples with duplicate ids (only necessary for multi-GPU evaluation)
    all_predictions = {pred["id"]: pred for pred in all_predictions}.values()

    assert len(all_predictions) == len(test_dataset)  # sanity check

    if dataset_name == "hateful_memes":
        # return ROC-AUC score
        gts = [pred["gt_label"] for pred in all_predictions]
        pred_scores = [pred["pred_score"] for pred in all_predictions]
        return roc_auc_score(gts, pred_scores)
    else:
        # return top-1 accuracy
        acc1 = sum(
            int(pred["gt_label"] == pred["pred_label"]) for pred in all_predictions
        )
        return float(acc1) / len(all_predictions)
