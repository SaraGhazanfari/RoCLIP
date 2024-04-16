# adapted from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/eval/evaluate.py
import argparse
import json
import os
import random
import time
import uuid
from collections import defaultdict

import numpy as np

from utils.captioning_utils import postprocess_captioning_generation, compute_cider

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
parser.add_argument(
    "--verbose", type=bool, default=True, help="Print the logs"
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


def finetune_clip(eval_model, data, args, image_processor, tokenizer):
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
                tokenizer=tokenizer,
                image_processor=image_processor,
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


def get_caption_prompt(caption=None) -> str:
    return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"


def evaluate_captioning(
        args: argparse.Namespace,
        data,
        model_args: dict,
        eval_model,
        tokenizer,
        image_processor,
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
    test_dataloader = data['val'].dataloader

    # in_context_samples = get_query_set(data['train'], args.query_set_size, seed, args)

    np.random.seed(seed)

    gt_dict = {}  # saves which gt works best for each image
    batch_n = 0
    for batch in data['train'].dataloader:
        print(len(batch[1]))

        batch_images = batch[0]
        batch_text = []
        for i in range(len(batch[0])):
            context_text = ""
            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            if effective_num_shots > 0:
                batch_text.append(context_text + get_caption_prompt())
            else:
                batch_text.append(get_caption_prompt())

        vision_x = batch_images.unsqueeze(1).unsqueeze(1)

        lang_x = tokenizer(batch_text, return_tensors="pt")
        start_time = time.time()
        outputs = eval_model.generate(vision_x=vision_x, lang_x=lang_x["input_ids"],
                                      attention_mask=lang_x["attention_mask"], max_new_tokens=20, num_beams=1)
                                      # output_scores=True, return_dict_in_generate=True)

        new_predictions = [
            tokenizer.decode(out) for out in outputs
        ]
        end_time = time.time()
        print('--------------------------------PRED--------------------------------')
        print(new_predictions)
        print('--------------------------------CORR--------------------------------')
        print(batch[1].tokens)
        print(f'{(end_time - start_time) / 60} minutes passed for {args.batch_size} batch size')
        # if batch_n < 20 and args.verbose:
        #     for k in range(len(new_predictions)):
        #         print(f"[gt] {batch[0][k]} [pred] {new_predictions[k]}")
        #     print(flush=True)

        uid = uuid.uuid4()
        results_path = f"{dataset_name}results_{uid}.json"
        results_path = os.path.join("captions-json", results_path)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        print(f"Saving generated captions to {results_path}")

        batch_n += 1

    with open(f'{os.path.dirname(args.results_file)}/gt_dict.json', 'w') as f:
        json.dump(gt_dict, f)

    metrics = compute_cider(
        result_path=results_path,
        annotations_path=args.coco_annotations_json_path
        if dataset_name == "coco"
        else args.flickr_annotations_json_path,
    )

    res = {"cider": metrics["CIDEr"] * 100.0, "success_rate": 0}
    return res, results_path
