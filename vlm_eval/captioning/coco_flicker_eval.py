import argparse
import json
import os
import uuid
from collections import defaultdict

from tqdm import tqdm

from open_flamingo.eval.coco_metric import (
    compute_cider,
    compute_cider_all_scores,
    postprocess_captioning_generation,
)
from open_flamingo.eval.eval_datasets import (
    CaptionDataset,
    TensorCaptionDataset,
)
from open_flamingo.eval.eval_model import BaseEvalModel
from vlm_eval.attacks.apgd import APGD
from vlm_eval.utils import *


def evaluate_captioning(
        args: argparse.Namespace,
        model_args: dict,
        eval_model: BaseEvalModel,
        seed: int = 42,
        min_generation_length: int = 0,
        max_generation_length: int = 20,
        num_beams: int = 3,
        length_penalty: float = -2.0,
        num_shots: int = 8,
        dataset_name: str = "coco",
        attack_config: dict = None,
):
    """Evaluate a model on COCO dataset.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 20.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        dataset_name (str, optional): dataset to evaluate on. Can be "coco" or "flickr". Defaults to "coco".
    Returns:
        float: CIDEr score

    """

    if dataset_name == "coco":
        image_train_dir_path = args.coco_train_image_dir_path
        image_val_dir_path = args.coco_val_image_dir_path
        annotations_path = args.coco_karpathy_json_path
    elif dataset_name == "flickr":
        image_train_dir_path = (
            args.flickr_image_dir_path
        )  # Note: calling this "train" for consistency with COCO but Flickr only has one split for images
        image_val_dir_path = None
        annotations_path = args.flickr_karpathy_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_dataset = CaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=True,
        dataset_name=dataset_name if dataset_name != "nocaps" else "coco",
    )

    test_dataset = CaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=False,
        dataset_name=dataset_name,
    )
    if args.from_saved:
        assert (
                dataset_name == "coco"
        ), "only coco supported for loading saved images, see TensorCaptionDataset"
        perturbation_dataset = TensorCaptionDataset(
            image_train_dir_path=image_train_dir_path,
            image_val_dir_path=args.from_saved,
            annotations_path=annotations_path,
            is_train=False,
            dataset_name=dataset_name,
        )

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
        seed,
    )

    in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)

    # attack stuff
    attack_str = attack_config["attack_str"]
    targeted = attack_config["targeted"]
    target_str = attack_config["target_str"]
    if attack_str != "none":
        mask_out = attack_config["mask_out"]
        if attack_config["save_adv"]:
            images_save_path = os.path.join(os.path.dirname(args.results_file), "adv-images")
            os.makedirs(images_save_path, exist_ok=True)
            print(f"saving adv images to {images_save_path}")
        if num_shots == 0:
            mask_out = None

    predictions = defaultdict()
    np.random.seed(seed)

    if attack_str == "ensemble":
        attacks = [
            (None, "float16", "clean", 0),
            ("apgd", "float16", "clean", 0),
            ("apgd", "float16", "clean", 1), ("apgd", "float16", "clean", 2),
            ("apgd", "float16", "clean", 3), ("apgd", "float16", "clean", 4),
            ("apgd", "float32", "prev-best", "prev-best")
        ]
    else:
        attacks = [(attack_str, 'none', 'clean', 0)]
    print(f"attacks: {attacks}")

    left_to_attack = {x["image_id"][0]: True for x in test_dataloader}  # hardcoded to batch size 1
    scores_dict = {x["image_id"][0]: np.inf for x in test_dataloader}  # hardcoded to batch size 1
    adv_images_dict = {}
    gt_dict = {}  # saves which gt works best for each image
    captions_attack_dict = {}  # saves the captions path for each attack
    captions_best_dict = {x["image_id"][0]: None for x in
                          test_dataloader}  # saves the best captions path for each image
    for attack_n, (attack_str_cur, precision, init, gt) in enumerate(attacks):
        print(f"attack_str_cur: {attack_str_cur}, precision: {precision}, init: {init}, gt: {gt}")
        test_dataset.which_gt = gt_dict if gt == "prev-best" else gt
        adv_images_cur_dict = {}
        if attack_n > 0 and attacks[attack_n - 1][1] != precision:
            # reload model with single precision
            device_id = eval_model.device
            ds_name = eval_model.dataset_name
            model_args["precision"] = precision
            eval_model.set_device("cpu")
            del eval_model
            torch.cuda.empty_cache()
            eval_model = get_eval_model(args, model_args, adversarial=True)
            eval_model.set_device(device_id)
            eval_model.dataset_name = ds_name
        for batch_n, batch in enumerate(tqdm(test_dataloader, desc=f"Running inference {dataset_name.upper()}")):
            if not left_to_attack[batch["image_id"][0]]:  # hardcoded to batch size 1
                continue

            batch_demo_samples = sample_batch_demos_from_query_set(
                in_context_samples, effective_num_shots, len(batch["image"])
            )
            batch_images = []
            batch_text = []
            batch_text_adv = []
            for i in range(len(batch["image"])):
                if num_shots > 0:
                    context_images = [x["image"] for x in batch_demo_samples[i]]
                else:
                    context_images = []
                batch_images.append(context_images + [batch["image"][i]])

                context_text = "".join(
                    [eval_model.get_caption_prompt(caption=x["caption"].strip()) for x in batch_demo_samples[i]]
                )

                print('**********************************')
                print(batch_demo_samples)
                print('**********************************')
                print(context_text)
                print('**********************************')
                # Keep the text but remove the image tags for the zero-shot case
                if num_shots == 0:
                    context_text = context_text.replace("<image>", "")

                adv_caption = batch["caption"][i] if not targeted else target_str
                if effective_num_shots > 0:
                    batch_text.append(context_text + eval_model.get_caption_prompt())
                    batch_text_adv.append(context_text + eval_model.get_caption_prompt(adv_caption))
                else:
                    batch_text.append(eval_model.get_caption_prompt())
                    batch_text_adv.append(eval_model.get_caption_prompt(adv_caption))

            batch_images = eval_model._prepare_images(batch_images)

            if args.from_saved:
                assert args.batch_size == 1
                assert init == "clean", "not implemented"
                # load the adversarial images, compute the perturbation
                # note when doing n-shot (n>0), have to make sure that context images
                # are the same as the ones where the perturbation was computed on
                adv = perturbation_dataset.get_from_id(batch["image_id"][0])
                # make sure adv has the same shape as batch_images
                if len(batch_images.shape) - len(adv.shape) == 1:
                    adv = adv.unsqueeze(0)
                elif len(batch_images.shape) - len(adv.shape) == -1:
                    adv = adv.squeeze(0)
                pert = adv - batch_images
                if attack_str_cur in [None, "none", "None"]:
                    # apply perturbation, otherwise it is applied by the attack
                    batch_images = batch_images + pert
            elif init == "prev-best":
                adv = adv_images_dict[batch["image_id"][0]].unsqueeze(0)
                pert = adv - batch_images
            else:
                assert init == "clean"
                pert = None

            ### adversarial attack
            if attack_str_cur not in [None, "none", "None"]:
                assert attack_str_cur == "apgd"
                eval_model.set_inputs(
                    batch_text=batch_text_adv,
                    past_key_values=None,
                    to_device=True,
                )
            if attack_str_cur == "apgd":
                # assert num_shots == 0
                attack = APGD(
                    eval_model if not targeted else lambda x: -eval_model(x),
                    norm="linf",
                    eps=attack_config["eps"],
                    mask_out=mask_out,
                    initial_stepsize=1.0,
                )
                batch_images = attack.perturb(
                    batch_images.to(eval_model.device, dtype=eval_model.cast_dtype),
                    iterations=attack_config["steps"],
                    pert_init=pert.to(eval_model.device, dtype=eval_model.cast_dtype) if pert is not None else None,
                    verbose=args.verbose if batch_n < 10 else False,
                )
                batch_images = batch_images.detach().cpu()
                ### end adversarial attack
            for i in range(batch_images.shape[0]):
                # save the adversarial images
                img_id = batch["image_id"][i]
                adv_images_cur_dict[img_id] = batch_images[i]

            outputs = eval_model.get_outputs(
                batch_images=batch_images,
                batch_text=batch_text,
                min_generation_length=min_generation_length,
                max_generation_length=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )

            new_predictions = [
                postprocess_captioning_generation(out).replace('"', "") for out in outputs
            ]
            if batch_n < 20 and args.verbose:
                for k in range(len(new_predictions)):
                    print(f"[gt] {batch['caption'][k]} [pred] {new_predictions[k]}")
                print(flush=True)
                # print(f"gt captions: {batch['caption']}")
                # print(f"new_predictions: {new_predictions}\n", flush=True)
            for i, sample_id in enumerate(batch["image_id"]):
                predictions[sample_id] = {"caption": new_predictions[i]}

        # save the predictions to a temporary file
        uid = uuid.uuid4()
        results_path = f"{dataset_name}results_{uid}.json"
        results_path = os.path.join(args.out_base_path, "captions-json", results_path)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        print(f"Saving generated captions to {results_path}")
        captions_attack_dict[f"{attack_str_cur}-{precision}-{init}-{gt}"] = results_path
        with open(results_path, "w") as f:
            f.write(
                json.dumps([{"image_id": k, "caption": predictions[k]["caption"]} for k in predictions], indent=4)
            )

        if attack_str == "ensemble":
            ciders, img_ids = compute_cider_all_scores(
                result_path=results_path,
                annotations_path=args.coco_annotations_json_path
                if dataset_name == "coco"
                else args.flickr_annotations_json_path,
                return_img_ids=True,
            )
            # if cider improved, save the new predictions
            # and if it is below thresh, set left to attack to false
            for cid, img_id in zip(ciders, img_ids):
                if cid < scores_dict[img_id]:
                    scores_dict[img_id] = cid
                    captions_best_dict[img_id] = predictions[img_id]["caption"]
                    adv_images_dict[img_id] = adv_images_cur_dict[img_id]
                    if isinstance(gt, int):
                        gt_dict.update({img_id: gt})
                cider_threshold = {"coco": 10., "flickr": 2.}[dataset_name]
                if cid < cider_threshold:
                    left_to_attack[img_id] = False
            # delete the temporary file
            # os.remove(results_path)
            # output how many left to attack
            n_left = sum(left_to_attack.values())
            print(f"##### "
                  f"after {(attack_str_cur, precision, gt)} left to attack: {n_left} "
                  f"current cider: {np.mean(ciders)}, best cider: {np.mean(list(scores_dict.values()))} "
                  f"cider-thresh: {cider_threshold}\n", flush=True)
            if n_left == 0:
                break
        else:
            adv_images_dict = adv_images_cur_dict

    if attack_config["save_adv"]:
        for img_id in adv_images_dict:
            torch.save(adv_images_dict[img_id], f'{images_save_path}/{str(img_id).zfill(12)}.pt')
    # save gt dict and left to attack dict
    with open(f'{os.path.dirname(args.results_file)}/gt_dict.json', 'w') as f:
        json.dump(gt_dict, f)
    with open(f'{os.path.dirname(args.results_file)}/left_to_attack.json', 'w') as f:
        json.dump(left_to_attack, f)
    with open(f'{os.path.dirname(args.results_file)}/captions_attack_dict.json', 'w') as f:
        json.dump(captions_attack_dict, f)

    if attack_str == "ensemble":
        assert None not in captions_best_dict.values()
        results_path = f"{dataset_name}results-best_{uuid.uuid4()}.json"
        results_path = os.path.join(args.out_base_path, "captions-json", results_path)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        print(f"Saving **best** generated captions to {results_path}")
        with open(results_path, "w") as f:
            f.write(
                json.dumps([{"image_id": k, "caption": captions_best_dict[k]} for k in captions_best_dict], indent=4)
            )

    metrics = compute_cider(
        result_path=results_path,
        annotations_path=args.coco_annotations_json_path
        if dataset_name == "coco"
        else args.flickr_annotations_json_path,
    )
    # delete the temporary file
    # os.remove(results_path)
    if not targeted:
        attack_success = np.nan
    else:
        attack_success = get_attack_success_rate(predictions, target_str)
    res = {"cider": metrics["CIDEr"] * 100.0, "success_rate": attack_success}
    return res, results_path
