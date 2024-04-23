import argparse
import json
import os
import uuid
from collections import defaultdict

from tqdm import tqdm

from data.llava_train_dataset import CC3MDataset
from open_flamingo.eval.eval_datasets import VQADataset
from open_flamingo.eval.eval_model import BaseEvalModel
from open_flamingo.eval.ok_vqa_utils import postprocess_ok_vqa_generation
from open_flamingo.eval.vqa_metric import (
    compute_vqa_accuracy,
    postprocess_vqa_generation,
)
from vlm_eval.attacks.apgd import APGD
from vlm_eval.utils import *


def evaluate_vqa(
        args: argparse.Namespace,
        model_args: dict,
        eval_model: BaseEvalModel,
        seed: int = 42,
        min_generation_length: int = 0,
        max_generation_length: int = 5,
        num_beams: int = 3,
        length_penalty: float = 0.0,
        num_shots: int = 8,
        dataset_name: str = "vqav2",
        attack_config: dict = None,
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0, OK-VQA, VizWiz and TextVQA.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
    Returns:
        float: accuracy score
    """

    if dataset_name == "ok_vqa":
        train_image_dir_path = args.ok_vqa_train_image_dir_path
        train_questions_json_path = args.ok_vqa_train_questions_json_path
        train_annotations_json_path = args.ok_vqa_train_annotations_json_path
        test_image_dir_path = args.ok_vqa_test_image_dir_path
        test_questions_json_path = args.ok_vqa_test_questions_json_path
        test_annotations_json_path = args.ok_vqa_test_annotations_json_path
    elif dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path
        test_questions_json_path = args.vqav2_test_questions_json_path
        test_annotations_json_path = args.vqav2_test_annotations_json_path
    elif dataset_name == "vizwiz":
        train_image_dir_path = args.vizwiz_train_image_dir_path
        train_questions_json_path = args.vizwiz_train_questions_json_path
        train_annotations_json_path = args.vizwiz_train_annotations_json_path
        test_image_dir_path = args.vizwiz_test_image_dir_path
        test_questions_json_path = args.vizwiz_test_questions_json_path
        test_annotations_json_path = args.vizwiz_test_annotations_json_path
    elif dataset_name == "textvqa":
        train_image_dir_path = args.textvqa_image_dir_path
        train_questions_json_path = args.textvqa_train_questions_json_path
        train_annotations_json_path = args.textvqa_train_annotations_json_path
        test_image_dir_path = args.textvqa_image_dir_path
        test_questions_json_path = args.textvqa_test_questions_json_path
        test_annotations_json_path = args.textvqa_test_annotations_json_path
    elif dataset_name != 'cc3m':
        raise ValueError(f"Unsupported dataset: {dataset_name}")


    if dataset_name != 'cc3m':
        train_dataset = VQADataset(
            image_dir_path=train_image_dir_path,
            question_path=train_questions_json_path,
            annotations_path=train_annotations_json_path,
            is_train=True,
            dataset_name=dataset_name,
        )

        test_dataset = VQADataset(
            image_dir_path=test_image_dir_path,
            question_path=test_questions_json_path,
            annotations_path=test_annotations_json_path,
            is_train=False,
            dataset_name=dataset_name,
        )
        if args.from_saved:
            perturbation_dataset = VQADataset(
                image_dir_path=args.from_saved,
                question_path=test_questions_json_path,
                annotations_path=test_annotations_json_path,
                is_train=False,
                dataset_name=dataset_name,
                is_tensor=True
            )
    else:
        test_questions_json_path = args.cc_test_questions_json_path
        test_annotations_json_path = args.cc_annotations_json_path
        train_dataset = CC3MDataset(
            img_root=args.train_data,
            annotations_path=args.cc_annotations_json_path,
        )

        test_dataset = CC3MDataset(
            img_root=args.train_data,
            annotations_path=args.cc_annotations_json_path,
        )
    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataloader = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
        seed,
    )

    in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)
    predictions = defaultdict()

    # attack stuff
    attack_str = attack_config["attack_str"]
    targeted = attack_config["targeted"]
    target_str = attack_config["target_str"]
    if attack_str != "none":
        target_str = attack_config["target_str"]
        mask_out = attack_config["mask_out"]
        eps = attack_config["eps"]
        if attack_config["save_adv"]:
            images_save_path = os.path.join(os.path.dirname(args.results_file), "adv-images")
            os.makedirs(images_save_path, exist_ok=True)
            print(f"saving adv images to {images_save_path}")
        if num_shots == 0:
            mask_out = None

    def get_sample_answer(answers):
        if len(answers) == 1:
            return answers[0]
        else:
            raise NotImplementedError

    np.random.seed(seed)

    if attack_str == "ensemble":
        attacks = [
            (None, "float16", "clean", 0), ("apgd", "float16", "clean", 0),
            ("apgd", "float16", "clean", 1), ("apgd", "float16", "clean", 2),
            ("apgd", "float16", "clean", 3), ("apgd", "float16", "clean", 4),
            ("apgd", "float32", "prev-best", "prev-best"),
            ("apgd-maybe", "float32", "clean", 0), ("apgd-Word", "float32", "clean", 0),
        ]
    else:
        attacks = [(attack_str, 'none', 'clean', 0)]
    print(f"attacks: {attacks}")

    left_to_attack = {x["question_id"][0]: True for x in test_dataloader}  # hardcoded to batch size 1
    scores_dict = {x["question_id"][0]: np.inf for x in test_dataloader}  # hardcoded to batch size 1
    adv_images_dict = {}
    gt_dict = {}  # saves which gt works best for each image
    answers_attack_dict = {}  # saves the captions path for each attack
    answers_best_dict = {x["question_id"][0]: None for x in
                         test_dataloader}  # saves the best captions path for each image
    for attack_n, (attack_str_cur, precision, init, gt) in enumerate(attacks):
        print(f"attack_str_cur: {attack_str_cur}, precision: {precision}, init: {init}, gt: {gt}")
        test_dataset.which_gt = gt_dict if gt == "prev-best" else gt
        adv_images_cur_dict = {}
        # if precision changed
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
        if attack_str_cur and "-" in attack_str_cur:
            targeted = True
            attack_str_cur, target_str = attack_str_cur.split("-")

        for batch_n, batch in enumerate(tqdm(test_dataloader, desc=f"Running inference {dataset_name}")):
            batch_demo_samples = sample_batch_demos_from_query_set(
                in_context_samples, effective_num_shots, len(batch["image"])
            )
            if not left_to_attack[batch["question_id"][0]]:  # hardcoded to batch size 1
                continue
            if len(batch['answers'][0]) == 0:  # hardcoded to batch size 1
                continue

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
                    [
                        eval_model.get_vqa_prompt(question=x["question"], answer=x["answers"][0])
                        for x in batch_demo_samples[i]
                    ]
                )

                # Keep the text but remove the image tags for the zero-shot case
                if num_shots == 0:
                    context_text = context_text.replace("<image>", "")

                adv_ans = get_sample_answer(batch["answers"][i]) if not targeted else target_str

                if effective_num_shots > 0:
                    batch_text.append(
                        context_text + eval_model.get_vqa_prompt(question=batch["question"][i])
                    )
                    batch_text_adv.append(
                        context_text + eval_model.get_vqa_prompt(question=batch["question"][i], answer=adv_ans)
                    )
                else:
                    batch_text.append(
                        eval_model.get_vqa_prompt(question=batch["question"][i])
                    )
                    batch_text_adv.append(
                        eval_model.get_vqa_prompt(question=batch["question"][i], answer=adv_ans)
                    )

            batch_images = eval_model._prepare_images(batch_images)

            if args.from_saved:
                assert args.batch_size == 1
                assert init == "clean", "not implemented"
                adv = perturbation_dataset.get_from_id(batch["question_id"][0]).unsqueeze(0)
                pert = adv - batch_images
                if attack_str_cur in [None, "none", "None"]:
                    # apply perturbation, otherwise it is applied by the attack
                    batch_images = batch_images + pert
            elif init == "prev-best":
                adv = adv_images_dict[batch["question_id"][0]].unsqueeze(0)
                pert = adv - batch_images
            else:
                assert init == "clean"
                pert = None

            ### adversarial attack
            if attack_str_cur == "apgd":
                eval_model.set_inputs(
                    batch_text=batch_text_adv,
                    past_key_values=None,
                    to_device=True,
                )
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
                q_id = batch["question_id"][i]
                adv_images_cur_dict[q_id] = batch_images[i]
            print(batch_text)
            outputs = eval_model.get_outputs(
                batch_images=batch_images,
                batch_text=batch_text,
                min_generation_length=min_generation_length,
                max_generation_length=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )
            print(outputs)
            process_function = (
                postprocess_ok_vqa_generation
                if dataset_name == "ok_vqa"
                else postprocess_vqa_generation
            )

            new_predictions = map(process_function, outputs)

            for new_prediction, sample_id in zip(new_predictions, batch["question_id"]):
                # predictions.append({"answer": new_prediction, "question_id": sample_id})
                predictions[sample_id] = new_prediction

            if batch_n < 20 and args.verbose:
                print(f"gt answer: {batch['answers']}")
                print(f"batch_text_adv: {batch_text_adv}")
                print(f"new_predictions: {[predictions[q_id] for q_id in batch['question_id']]}\n", flush=True)

        # save the predictions to a temporary file
        random_uuid = str(uuid.uuid4())
        results_path = f"{dataset_name}results_{random_uuid}.json"
        results_path = os.path.join(args.out_base_path, "captions-json", results_path)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        print(f"Saving generated captions to {results_path}")
        answers_attack_dict[f"{attack_str_cur}-{precision}-{init}-{gt}"] = results_path
        with open(results_path, "w") as f:
            f.write(json.dumps([{"answer": predictions[k], "question_id": k} for k in predictions], indent=4))

        if attack_str == "ensemble":
            acc_dict_cur = compute_vqa_accuracy(
                results_path,
                test_questions_json_path,
                test_annotations_json_path,
                return_individual_scores=True
            )
            for q_id, pred in predictions.items():
                acc = acc_dict_cur[q_id]
                if acc < scores_dict[q_id]:
                    scores_dict[q_id] = acc
                    answers_best_dict[q_id] = pred
                    adv_images_dict[q_id] = adv_images_cur_dict[q_id]
                    if isinstance(gt, int):
                        gt_dict.update({q_id: gt})
                if acc == 0.:
                    left_to_attack[q_id] = False
            print(
                f"##### "
                f"after {(attack_str_cur, precision, gt)} left to attack: {sum(left_to_attack.values())} "
                f"current acc: {np.mean(list(acc_dict_cur.values()))}, best acc: {np.mean(list(scores_dict.values()))}\n",
                flush=True
            )

    if attack_config["save_adv"]:
        for q_id in adv_images_dict:
            torch.save(adv_images_dict[q_id], f'{images_save_path}/{str(q_id).zfill(12)}.pt')
    # save gt dict and left to attack dict
    with open(f'{os.path.dirname(args.results_file)}/gt_dict.json', 'w') as f:
        json.dump(gt_dict, f)
    with open(f'{os.path.dirname(args.results_file)}/left_to_attack.json', 'w') as f:
        json.dump(left_to_attack, f)
    with open(f'{os.path.dirname(args.results_file)}/captions_attack_dict.json', 'w') as f:
        json.dump(answers_attack_dict, f)

    if attack_str == "ensemble":
        assert None not in answers_best_dict.values()
        results_path = f"{dataset_name}results-best_{uuid.uuid4()}.json"
        results_path = os.path.join(args.out_base_path, "captions-json", results_path)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        print(f"Saving **best** generated captions to {results_path}")
        answers_best_list = [{"answer": answers_best_dict[k], "question_id": k} for k in answers_best_dict]
        with open(results_path, "w") as f:
            f.write(json.dumps(answers_best_list, indent=4))

    acc = compute_vqa_accuracy(
        results_path,
        test_questions_json_path,
        test_annotations_json_path,
        dataset=dataset_name
    )

    return acc, results_path
