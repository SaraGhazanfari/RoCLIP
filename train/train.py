# adapted from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/eval/evaluate.py
import argparse
import json
import os
import random
import time
import uuid
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from utils.captioning_utils import postprocess_captioning_generation, compute_cider, compute_cider_all_scores

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model", type=str, default='open_flamingo', help="VLM model name"
)
parser.add_argument(
    "--query_set_size", type=int, default=2048, help="Size of demonstration query set"
)
parser.add_argument(
    "--results_file", type=str, default='./out', help="JSON file to save results"
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0], type=int)  # todo [0, 4, 8, 16, 32]
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=1000,
    help="Number of samples to evaluate on. -1 for all samples.",
)


def get_result_file_name_and_dir(args):
    results_file_dir = f"{args.results_file}_{'_'.join(['sbucaptions'])}"
    results_file_dir += f"_{args.num_samples}samples"
    tme = time.strftime("%Y-%m-%d_%H-%M-%S")
    results_file_dir += f"_{tme}"
    results_file_dir = os.path.join('results', results_file_dir)
    results_file_name = os.path.join(results_file_dir, 'results.json')
    return results_file_dir, results_file_name


def finetune_clip(eval_model, data, args):
    finetune_args, leftovers = parser.parse_known_args()

    args = argparse.Namespace(**{**vars(args), **vars(finetune_args)})
    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }

    results = defaultdict(list)

    results["model"] = leftovers
    results_file_dir, results_file_name = get_result_file_name_and_dir(args)
    for shot in args.shots:
        scores = {'cider': [], 'success_rate': []}
        for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
            res, out_captions_json = evaluate_captioning(
                args,
                data=data,
                model_args=model_args,
                eval_model=eval_model,
                num_shots=shot,
                seed=seed,
                dataset_name="coco"
            )
            print(f"Shots {shot} Trial {trial} Score: {res}")
            scores['cider'].append(res['cider'])
            scores['success_rate'].append(res['success_rate'])

        print(f"Shots {shot} Mean CIDEr score: {np.nanmean(scores['cider'])}")
        print(f"Shots {shot} Mean Success rate: {np.nanmean(scores['success_rate'])}")
        results["coco"].append(
            {
                "shots": shot,
                "trials": scores,
                "mean": {'cider': np.nanmean(scores['cider']), 'success_rate': np.nanmean(scores['success_rate'])},
                "captions": out_captions_json,
            }
        )
    if args.results_file is not None:
        with open(results_file_name, "w") as f:
            json.dump(results, f)
    del res, out_captions_json


def get_query_set(train_dataset, query_set_size, seed, args):
    np.random.seed(seed)
    query_set = np.random.choice(args.train_num_samples, query_set_size, replace=False)
    return [train_dataset[i] for i in query_set]


def prepare_eval_samples(test_dataset, num_samples, batch_size, seed, args):
    np.random.seed(seed)
    # TODO
    random_indices = np.random.choice(1000, num_samples, replace=False)
    dataset = torch.utils.data.Subset(test_dataset, random_indices)
    sampler = torch.utils.data.SequentialSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )
    return loader


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]


def compute_effective_num_shots(num_shots, model_type):
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def get_attack_success_rate(predictions, target_str):
    n_success = 0
    n = 0
    for k in predictions:
        n += 1
        caption = predictions[k]["caption"]
        # check if target_str is contained in caption
        if target_str.lower() in caption.lower():
            n_success += 1
    return n_success / n * 100


def evaluate_captioning(
        args: argparse.Namespace,
        data,
        model_args: dict,
        eval_model,
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

    effective_num_shots = compute_effective_num_shots(num_shots, args.model)

    test_dataloader = prepare_eval_samples(
        data['val'],
        args.num_samples if args.num_samples > 0 else len(data['val'][1]),
        args.batch_size,
        seed,
        args,
    )

    # in_context_samples = get_query_set(data['train'], args.query_set_size, seed, args)

    predictions = defaultdict()
    np.random.seed(seed)

    left_to_attack = {x["image_id"][0]: True for x in test_dataloader}  # hardcoded to batch size 1
    scores_dict = {x["image_id"][0]: np.inf for x in test_dataloader}  # hardcoded to batch size 1
    adv_images_dict = {}
    gt_dict = {}  # saves which gt works best for each image
    captions_attack_dict = {}  # saves the captions path for each attack
    captions_best_dict = {x["image_id"][0]: None for x in
                          test_dataloader}  # saves the best captions path for each image

    for batch_n, batch in enumerate(tqdm(test_dataloader, desc=f"Running inference {dataset_name.upper()}")):
        if not left_to_attack[batch["image_id"][0]]:  # hardcoded to batch size 1
            continue

        # batch_demo_samples = sample_batch_demos_from_query_set(
        #     in_context_samples, effective_num_shots, len(batch["image"])
        # )
        batch_images = []
        batch_text = []
        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch["image"][i]])

            context_text = "".join(
                [eval_model.get_caption_prompt(caption=x["caption"].strip()) for x in batch_demo_samples[i]]
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")
            print('----------------------------------------------')
            print(context_text)
            print('----------------------------------------------')

            if effective_num_shots > 0:
                batch_text.append(context_text + eval_model.get_caption_prompt())
            else:
                batch_text.append(eval_model.get_caption_prompt())

        batch_images = eval_model._prepare_images(batch_images)

        for i in range(batch_images.shape[0]):
            # save the adversarial images
            img_id = batch["image_id"][i]

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
        results_path = os.path.join("captions-json", results_path)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        print(f"Saving generated captions to {results_path}")
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

                cider_threshold = {"coco": 10., "flickr": 2.}[dataset_name]
                if cid < cider_threshold:
                    left_to_attack[img_id] = False
            # delete the temporary file
            # os.remove(results_path)
            # output how many left to attack
            n_left = sum(left_to_attack.values())
            print(f"##### "
                  f"current cider: {np.mean(ciders)}, best cider: {np.mean(list(scores_dict.values()))} "
                  f"cider-thresh: {cider_threshold}\n", flush=True)
            if n_left == 0:
                break

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
