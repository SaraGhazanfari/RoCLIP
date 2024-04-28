import random

import numpy as np
import torch


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples + query_set_size must be less than {len(full_dataset)}"
        )

    # get a random subset of the dataset
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


def force_cudnn_initialization():
    # https://stackoverflow.com/questions/66588715/runtimeerror-cudnn-error-cudnn-status-not-initialized-using-pytorch
    s = 32
    dev = torch.device("cuda:0")
    torch.nn.functional.conv2d(
        torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev)
    )


def get_eval_model(args, model_args, adversarial):
    from open_flamingo.eval.models.llava import EvalModelLLAVA
    from open_flamingo.eval.models.of_eval_model_adv import EvalModelAdv
    if args.model == "open_flamingo":
        return EvalModelAdv(model_args, adversarial=adversarial)
    elif args.model == "llava":
        return EvalModelLLAVA(model_args)
    else:
        raise ValueError(f"Unsupported model: {args.model}")


def get_query_set(train_dataset, query_set_size, seed):
    np.random.seed(seed)
    query_set = np.random.choice(len(train_dataset), query_set_size, replace=False)
    return [train_dataset[i] for i in query_set]


def prepare_eval_samples(test_dataset, num_samples, batch_size, seed, len_test_dataset=0):
    np.random.seed(seed)
    len_test_dataset = len_test_dataset if len_test_dataset > 0 else len(test_dataset)
    random_indices = np.random.choice(len_test_dataset, num_samples, replace=False)
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
