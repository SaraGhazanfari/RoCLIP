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
    dev = torch.device(f"cuda:0")
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


class TinyLLAVA:
    def __init__(self, args, device):
        from transformers import LlavaForConditionalGeneration
        self.conv_mode = "vicuna_v1"
        args.model_path = "bczhou/tiny-llava-v1-hf"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        kwargs = {}
        if args.precision == 'float16':
            kwargs['torch_dtype'] = torch.float16
        elif args.precision == 'float32':
            kwargs['torch_dtype'] = torch.float32
        processor = AutoProcessor.from_pretrained(args.model_path)
        self.image_processor = processor.image_processor
        self._prepare_tokenizer(processor)
        self.config = AutoConfig.from_pretrained(args.model_path)
        setattr(self.config, 'image_aspect_ratio', 'pad')
        self.mm_use_im_start_end = getattr(self.config, "mm_use_im_start_end", False)

    def _prepare_tokenizer(self, processor):
        self.tokenizer = processor.tokenizer
        mm_use_im_start_end = getattr(self.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(self.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _prepare_images(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        assert len(batch) == 1, "Only support batch size 1 (yet)"
        image_tensor = process_images(batch[0], self.image_processor, self.config)
        return image_tensor

    def set_device(self, device):
        """Set device for model."""
        self.device = f"cuda:{device}"
        self.model = self.model.to(self.device)

    def _prepare_text(
            self,
            convs,
            past_key_values: torch.Tensor = None,
            to_device: bool = False,
    ):
        input_ids = [
            tokenizer_image_token(conv.get_prompt(), self.tokenizer, image_token_index=self.config.image_token_index,
                                  return_tensors='pt') for conv in convs
        ]
        input_ids = torch.stack(input_ids, dim=0)

        context_only = convs[0].get_prompt().split("ASSISTANT:")[0] + "ASSISTANT:"
        context_len = len(self.tokenizer.encode(context_only))

        labels = copy.deepcopy(input_ids)#[:, context_len:]
        labels[:, :context_len] = self.config.ignore_index
        attention_mask = input_ids.ne(self.config.pad_token_id)
        return input_ids[:, :context_len], labels, attention_mask, past_key_values

    def get_caption_prompt(self, caption=None) -> str:
        qs = "Provide a short caption for this image."

        if self.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], caption)

        return conv