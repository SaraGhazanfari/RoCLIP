# adapted from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/eval/evaluate.py
import json
import os
import time
from collections import defaultdict

from open_flamingo.eval.classification_utils import (
    TARGET_TO_SEED
)
from vlm_eval.captioning.captioning_eval import evaluate_captioning
from vlm_eval.captioning.vqa_eval import evaluate_vqa
from vlm_eval.param import parser
from vlm_eval.utils import *


def main():
    args, leftovers = parser.parse_known_args()
    if args.targeted:
        assert args.target_str is not None
        # set seed
        args.trial_seeds = TARGET_TO_SEED[f"{args.target_str}"]
    assert args.eps >= 1
    # set visible device
    if args.device_n is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_n)

    if args.mask_out != "none": assert args.model == "open_flamingo"
    attack_config = {
        "attack_str": args.attack,
        "eps": args.eps / 255,
        "steps": args.steps,
        "mask_out": args.mask_out,
        "targeted": args.targeted,
        "target_str": args.target_str,
        "from_saved": args.from_saved,
        "save_adv": (not args.dont_save_adv) and args.attack != "none",
    }

    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    print(f"Arguments:\n{'-' * 20}")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("\n### model args")
    for arg, value in model_args.items():
        print(f"{arg}: {value}")
    print(f"{'-' * 20}")
    print("Clean evaluation" if args.attack == "none" else "Adversarial evaluation")
    eval_model = get_eval_model(args, model_args, adversarial=attack_config["attack_str"] != "none")

    force_cudnn_initialization()

    device_id = 0
    eval_model.set_device(device_id)

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")
    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")
    if args.attack == "ensemble":
        assert model_args["precision"] == "float16"

    # create results file name
    eval_datasets_list = [
        'cc' if args.eval_cc else "",
        "sbu" if args.eval_sbu else "",
        "coco" if args.eval_coco else "",
        "vqav2" if args.eval_vqav2 else "",
        "ok_vqa" if args.eval_ok_vqa else "",
        "vizwiz" if args.eval_vizwiz else "",
        "textvqa" if args.eval_textvqa else "",
        "imagenet" if args.eval_imagenet else "",
        "flickr30" if args.eval_flickr30 else "",
    ]
    eval_datasets_list = [x for x in eval_datasets_list if x != ""]
    results_file_dir = f"{args.results_file}_{'_'.join(eval_datasets_list)}"
    if (v := eval_model.model_args.get("vision_encoder_pretrained")) is not None:
    #     v = ("-" + v.split("/")[-3]) if "/" in v else v
    #     if len(v) > 180:
    #         v = v[140:]
        results_file_dir += v
    if args.attack not in [None, "none"]:
        results_file_dir += f"_{args.attack}_{args.eps}_{args.steps}_{args.mask_out}_{''.join(map(str, args.shots))}-shot"
    if args.from_saved:
        results_file_dir += f"_FROM_{'-'.join(args.from_saved.split('/')[-2:])}"
    if args.targeted:
        results_file_dir += f"_targeted={args.target_str.replace(' ', '-').replace('/', '-')}"
    results_file_dir += f"_{args.num_samples}samples"
    tme = time.strftime("%Y-%m-%d_%H-%M-%S")
    results_file_dir += f"_{tme}"
    results_file_dir = os.path.join(args.out_base_path, 'results', results_file_dir)
    os.makedirs(results_file_dir, exist_ok=True)
    results_file_name = os.path.join(results_file_dir, 'results.json')
    args.results_file = results_file_name
    print(f"Results will be saved to {results_file_name}")
    results = defaultdict(list)
    # add model information to results
    results["model"] = leftovers
    results["attack"] = attack_config

    if args.eval_flickr30:
        print("Evaluating on Flickr30k...")
        eval_model.dataset_name = "flickr"
        for shot in args.shots:
            scores = {'cider': [], 'success_rate': []}
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                res, out_captions_json = evaluate_captioning(
                    args,
                    model_args=model_args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="flickr",
                    min_generation_length=0,
                    max_generation_length=20,
                    num_beams=3,
                    attack_config=attack_config,
                )
                print(f"Shots {shot} Trial {trial} Score: {res}")
                scores['cider'].append(res['cider'])
                scores['success_rate'].append(res['success_rate'])

            print(f"Shots {shot} Mean CIDEr score: {np.nanmean(scores['cider'])}")
            print(f"Shots {shot} Mean Success rate: {np.nanmean(scores['success_rate'])}")
            results["flickr30"].append(
                {
                    "shots": shot,
                    "trials": scores,
                    "mean": {
                        'cider': np.nanmean(scores['cider']),
                        'success_rate': np.nanmean(scores['success_rate'])
                    },
                    "captions": out_captions_json,
                }
            )
        if args.results_file is not None:
            with open(results_file_name, "w") as f:
                json.dump(results, f)
        del res, out_captions_json

    if args.eval_coco:
        print("Evaluating on COCO...")
        eval_model.dataset_name = "coco"
        for shot in args.shots:
            scores = {'cider': [], 'success_rate': []}
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                res, out_captions_json = evaluate_captioning(
                    args,
                    model_args=model_args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="coco",
                    attack_config=attack_config,
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

    if args.eval_sbu:
        print("Evaluating on SBUCaption...")
        eval_model.dataset_name = "sbu"
        for shot in args.shots:
            scores = {'cider': [], 'success_rate': []}
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                res, out_captions_json = evaluate_captioning(
                    args,
                    model_args=model_args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="sbu",
                    attack_config=attack_config,
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

    if args.eval_cc:
        print("Evaluating on COCO...")
        eval_model.dataset_name = "cc3m"
        for shot in args.shots:
            scores = {'cider': [], 'success_rate': []}
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                res, out_captions_json = evaluate_captioning(
                    args,
                    model_args=model_args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="cc3m",
                    attack_config=attack_config,
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
    # if args.eval_cc:
    #     print("Evaluating on CC-3M...")
    #     eval_model.dataset_name = "cc3m"
    #     for shot in args.shots:
    #         scores = []
    #         for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
    #             ok_vqa_score, out_captions_json = evaluate_vqa(
    #                 args=args,
    #                 model_args=model_args,
    #                 eval_model=eval_model,
    #                 num_shots=shot,
    #                 seed=seed,
    #                 dataset_name="cc3m",
    #                 attack_config=attack_config,
    #                 max_generation_length=20
    #             )
    #             print(f"Shots {shot} Trial {trial} OK-VQA score: {ok_vqa_score}")
    #             scores.append(ok_vqa_score)
    #
    #         print(f"Shots {shot} Mean OK-VQA score: {np.nanmean(scores)}")
    #         results["ok_vqa"].append(
    #             {
    #                 "shots": shot,
    #                 "trials": scores,
    #                 "mean": np.nanmean(scores),
    #                 "captions": out_captions_json,
    #             }
    #         )
    #     del ok_vqa_score, out_captions_json

    if args.eval_ok_vqa:
        print("Evaluating on OK-VQA...")
        eval_model.dataset_name = "ok_vqa"
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                ok_vqa_score, out_captions_json = evaluate_vqa(
                    args=args,
                    model_args=model_args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="ok_vqa",
                    attack_config=attack_config,
                )
                print(f"Shots {shot} Trial {trial} OK-VQA score: {ok_vqa_score}")
                scores.append(ok_vqa_score)

            print(f"Shots {shot} Mean OK-VQA score: {np.nanmean(scores)}")
            results["ok_vqa"].append(
                {
                    "shots": shot,
                    "trials": scores,
                    "mean": np.nanmean(scores),
                    "captions": out_captions_json,
                }
            )
        del ok_vqa_score, out_captions_json

    if args.eval_vqav2:
        print("Evaluating on VQAv2...")
        eval_model.dataset_name = "vqav2"
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vqa_score, out_captions_json = evaluate_vqa(
                    args=args,
                    model_args=model_args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vqav2",
                    attack_config=attack_config,
                )
                print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                scores.append(vqa_score)

            print(f"Shots {shot} Mean VQA score: {np.nanmean(scores)}")
            results["vqav2"].append(
                {
                    "shots": shot,
                    "trials": scores,
                    "mean": np.nanmean(scores),
                    "captions": out_captions_json,
                }
            )
        del vqa_score, out_captions_json

    if args.eval_vizwiz:
        print("Evaluating on VizWiz...")
        eval_model.dataset_name = "vizwiz"
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vizwiz_score, out_captions_json = evaluate_vqa(
                    args=args,
                    model_args=model_args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vizwiz",
                    attack_config=attack_config,
                )
                print(f"Shots {shot} Trial {trial} VizWiz score: {vizwiz_score}")
                scores.append(vizwiz_score)

            print(f"Shots {shot} Mean VizWiz score: {np.nanmean(scores)}")
            results["vizwiz"].append(
                {
                    "shots": shot,
                    "trials": scores,
                    "mean": np.nanmean(scores),
                    "captions": out_captions_json,
                }
            )
        del vizwiz_score, out_captions_json

    if args.eval_textvqa:
        print("Evaluating on TextVQA...")
        eval_model.dataset_name = "textvqa"
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                textvqa_score, out_captions_json = evaluate_vqa(
                    args=args,
                    model_args=model_args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="textvqa",
                    max_generation_length=10,
                    attack_config=attack_config,
                )
                print(f"Shots {shot} Trial {trial} TextVQA score: {textvqa_score}")
                scores.append(textvqa_score)

            print(f"Shots {shot} Mean TextVQA score: {np.nanmean(scores)}")
            results["textvqa"].append(
                {
                    "shots": shot,
                    "trials": scores,
                    "mean": np.nanmean(scores),
                    "captions": out_captions_json,
                }
            )
        del textvqa_score, out_captions_json

    if args.eval_imagenet:
        raise NotImplementedError
        print("Evaluating on ImageNet...")
        eval_model.dataset_name = "imagenet"
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                imagenet_score = evaluate_classification(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    no_kv_caching=args.no_caching_for_classification,
                    dataset_name="imagenet",
                    attack_config=attack_config,
                )
                print(
                    f"Shots {shot} Trial {trial} "
                    f"ImageNet score: {imagenet_score}"
                )
                scores.append(imagenet_score)

            print(f"Shots {shot} Mean ImageNet score: {np.nanmean(scores)}")
            results["imagenet"].append(
                {"shots": shot, "trials": scores, "mean": np.nanmean(scores)}
            )
        del imagenet_score

    if args.eval_hateful_memes:
        raise NotImplementedError
        print("Evaluating on Hateful Memes...")
        eval_model.dataset_name = "hateful_memes"
        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                hateful_memes_score, out_captions_json = evaluate_classification(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    no_kv_caching=args.no_caching_for_classification,
                    dataset_name="hateful_memes",
                    attack_config=attack_config,
                )
                print(
                    f"Shots {shot} Trial {trial} "
                    f"Hateful Memes score: {hateful_memes_score}"
                )
                scores.append(hateful_memes_score)

            print(f"Shots {shot} Mean Hateful Memes score: {np.nanmean(scores)}")
            results["hateful_memes"].append(
                {
                    "shots": shot,
                    "trials": scores,
                    "mean": np.nanmean(scores),
                    "captions": out_captions_json,
                }
            )
        del hateful_memes_score, out_captions_json

    if args.results_file is not None:
        with open(results_file_name, "w") as f:
            json.dump(results, f)
        print(f"Results saved to {results_file_name}")

    print("\n### model args")
    for arg, value in model_args.items():
        print(f"{arg}: {value}")
    print(f"{'-' * 20}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"Total time: {total_time // 3600}h {(total_time % 3600) // 60}m {total_time % 60:.0f}s")
