import copy
import sys
import time
from typing import List

from torch.nn import DataParallel
from transformers import AutoProcessor, AutoConfig

from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from train.datasets import COCOFlickrDataset
from train.pgd_train import pgd
from vlm_eval.attacks.apgd import apgd
from vlm_eval.utils import force_cudnn_initialization

# from vlm_eval.utils import get_eval_model

sys.path.append("open_flamingo")
import os
import shutil
import string
import random

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from training.scheduler import cosine_lr
import wandb
from train.utils import init_wandb
from open_flamingo.eval.models.utils import unwrap_model
from train.utils import str2bool

import argparse
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Model name. `open_flamingo` and `llava` supported.",
                    default="llava", choices=["open_flamingo", "llava", "tiny-llava"], )
parser.add_argument('--clip_model_name', type=str, default='ViT-L-14', help='ViT-L-14, ViT-B-32')
parser.add_argument('--pretrained', type=str, default='openai')
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--template', type=str, default='std')
parser.add_argument('--imagenet_root', type=str, default='/mnt/datasets/imagenet',
                    help='Imagenet dataset root directory')
parser.add_argument('--output_normalize', type=str2bool, default=False, help='Whether the embedding is normalized')
parser.add_argument('--start_step', type=int, default=0, help='Start step for training')
parser.add_argument('--optimizer_state', type=str, default='', help='Optimizer state file path')
parser.add_argument('--steps', type=int, default=20000, help='Number of training steps')
parser.add_argument('--warmup', type=int, default=14000, help='Warmup steps')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--loss', type=str, default='l2', help='ce, l2')
parser.add_argument('--loss_clean', type=str, default='none', help='ce, l2')
parser.add_argument('--clean_weight', type=float, default=0.0, help='Weight for clean loss')
parser.add_argument('--trades', type=str2bool, default=False, help='Use TRADES')
parser.add_argument('--opt', type=str, default='adamw', help='Optimizer type; sgd, adamw')
parser.add_argument('--momentum_sgd', type=float, default=0.9, help='Momentum for SGD optimizer')
parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--attack', type=str, default=None, help='Adversarial attack type')
parser.add_argument('--inner_loss', type=str, default='l2', help='Inner loss function for adversarial training')
parser.add_argument('--norm', type=str, default='linf', help='Norm for adversarial perturbation')
parser.add_argument('--eps', type=float, default=4, help='Epsilon for adversarial perturbation')
parser.add_argument('--iterations_adv', type=int, default=10, help='Iterations for adversarial attack')
parser.add_argument('--stepsize_adv', type=float, default=1.,
                    help='Step size for adversarial attack (no effect for apgd)')
parser.add_argument('--wandb', type=str2bool, default=True, help='Use Weights & Biases for logging')
parser.add_argument('--experiment_name', type=str, default='')
parser.add_argument('--overwrite', type=str2bool, default=False, help='Overwrite existing directory')
parser.add_argument('--log_freq', type=int, default=1, help='Logging frequency')
parser.add_argument('--eval_freq', type=int, default=50, help='Evaluation frequency')
parser.add_argument('--output_dir', type=str, default='', help='Output directory')
parser.add_argument('--save_checkpoints', type=str2bool, default=True, help='Save 10 training checkpoints')
parser.add_argument('--devices', type=str, default='', help='Device IDs for CUDA')
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
    distributed training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
# leftovers
parser.add_argument("--model_path", type=str)
parser.add_argument("--temperature", type=float)
parser.add_argument("--num_beams", type=int)
parser.add_argument("--precision", type=str)
parser.add_argument("--vision_encoder_pretrained", type=str)


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
        self.tokenizer = processor.tokenizer
        self.config = AutoConfig.from_pretrained(args.model_path)
        setattr(self.config, 'image_aspect_ratio', 'pad')
        self.mm_use_im_start_end = getattr(self.config, "mm_use_im_start_end", False)

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

        labels = copy.deepcopy(input_ids)
        labels[:, :context_len] = self.config.ignore_index
        attention_mask = input_ids.ne(self.config.pad_token_id)
        return input_ids, labels, attention_mask, past_key_values

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


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 8
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def main(args, leftovers):
    # setup wandb
    if args.wandb:
        init_wandb(
            project_name='clip-finetune',
            model_name=args.finetuned_model_name,
            config=vars(args)
        )
    else:
        wandb.init(mode='disabled')

    # print args
    print(f"Arguments:\n{'-' * 20}")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print(f"{'-' * 20}")

    # setup dirs
    if args.overwrite:
        shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=False)

    # write args to file
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    main_device = 0
    # get models
    model_orig, _, image_processor = open_clip.create_model_and_transforms(
        args.clip_model_name, pretrained='openai'
    )
    if args.optimizer_state != '':
        assert args.start_step > 0
        assert str(args.start_step) in args.optimizer_state
        assert args.pretrained in ['', 'none']
        args.pretrained = args.optimizer_state.replace('_opt', '')

    image_dir_path = f'{args.imagenet_root}/train2014'
    annotations_path = f'{args.imagenet_root}/annotations/captions_train2014.json'
    # model_args = {
    #     leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    # }
    model = TinyLLAVA(args, main_device)  # get_eval_model(args, args.__dict__, adversarial="none")
    dataset = COCOFlickrDataset(model=model,
                                image_processor=model._prepare_images,
                                image_dir_path=image_dir_path,
                                annotations_path=annotations_path,
                                transform=None,
                                prefix='COCO_train2014_'
                                )

    # init_distributed_mode(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    # dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    force_cudnn_initialization()
    device_id = 0
    model.set_device(device_id)
    params = model.model.vision_tower.vision_model.parameters()
    if num_gpus > 1:
        model = DataParallel(model.model, device_ids=range(num_gpus))
    else:
        model = model.model
    # set optimizer (all params have requires_grad=True)

    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum_sgd,
            weight_decay=args.wd
        )
    else:
        raise ValueError(f'Optimizer {args.optimizer} not supported.')
    if args.optimizer_state != '':
        optimizer.load_state_dict(torch.load(args.optimizer_state))

    # set scheduler
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, args.steps)

    # compute amount of epochs
    total_epochs = args.steps / len(dataloader)
    print(f'train for {total_epochs} epochs')
    args.total_epochs = total_epochs

    # finetune
    step_total = args.start_step
    epoch = 0
    while step_total < args.steps:
        train_one_epoch(model=model, dataloader=dataloader, args=args, optimizer=optimizer, scheduler=scheduler,
                        step_total=step_total)
        print(f'Epoch {epoch} done.')
        epoch += 1

    # save final model
    torch.save(unwrap_model(model).model.state_dict(), f'{args.output_dir}/checkpoints/final.pt')
    torch.save(optimizer.state_dict(), f'{args.output_dir}/checkpoints/final_opt.pt')

    if args.output_dir.endswith('_temp'):
        # rename temp dir to final dir
        os.rename(args.output_dir, args.output_dir[:-5])


class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, args, normalize):
        super().__init__()
        self.model = model
        self.args = args
        self.normalize = normalize

    def forward(self, vision, output_normalize):
        embedding = self.model(self.normalize(vision))
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)
        return embedding


class ComputeLossWrapper:
    def __init__(self, embedding_orig, embedding_text_labels_norm, reduction='mean', loss=None,
                 logit_scale=100.):
        self.embedding_orig = embedding_orig
        self.embedding_text_labels_norm = embedding_text_labels_norm
        self.reduction = reduction
        self.loss_str = loss
        self.logit_scale = logit_scale

    def __call__(self, embedding, targets):
        return compute_loss(
            loss_str=self.loss_str, embedding=embedding, targets=targets,
            embedding_orig=self.embedding_orig, logit_scale=self.logit_scale,
            embedding_text_labels_norm=self.embedding_text_labels_norm, reduction=self.reduction
        )


def train_one_epoch(model, dataloader, args, optimizer, scheduler, step_total):
    unwrap_model(model).vision_tower.vision_model.train()
    start_time, end_time = time.time(), time.time()

    for i, (data, input_ids, labels, attention_mask) in enumerate(dataloader):
        print(f'{i}/{len(dataloader)} Time:{round(end_time - start_time, 4)}')
        start_time = time.time()
        data, input_ids, labels, attention_mask = data.to('cuda:0'), input_ids.to('cuda:0'), labels.to(
            'cuda:0'), attention_mask.to('cuda:0')

        if args.attack == 'pgd':
            data_adv = pgd(
                forward=model,
                loss_fn=None,
                data_clean=data,
                targets=labels,
                norm=args.norm,
                eps=args.eps,
                iterations=args.iterations_adv,
                stepsize=args.stepsize_adv,
                output_normalize=args.output_normalize,
                perturbation=torch.zeros_like(data).uniform_(-args.eps, args.eps).requires_grad_(True),
                mode='max',
                verbose=True,
                input_ids=input_ids, labels=labels, attention_mask=attention_mask
            )
        elif args.attack == 'apgd':
            # apgd currently always applies output normalization
            data_adv = apgd(
                model=model,
                loss_fn=None,
                x=data,
                y=labels,
                norm=args.norm,
                eps=args.eps,
                n_iter=args.iterations_adv,
                verbose=True
            )
        elif not args.attack:
            data_adv = data
        if args.clean_weight > 0.:
            loss_clean = torch.mean(
                model(pixel_values=data, input_ids=input_ids, attention_mask=attention_mask, past_key_values=None,
                      inputs_embeds=None, labels=labels).loss)
        else:
            loss_clean = 0.
        print('3', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        out = model(pixel_values=data_adv, input_ids=input_ids, attention_mask=attention_mask, past_key_values=None,
                    inputs_embeds=None, labels=labels)
        print(out)
        loss = torch.mean(out.loss)
        print(f'$$$$$$$$$$$$$$$$$$loss: {loss}, loss_clean: {loss_clean}*****************************')
        loss_total = args.clean_weight * loss_clean + (1 - args.clean_weight) * loss
        loss_total.backward()
        optimizer.step()
        optimizer.zero_grad()
        step_total += 1
        scheduler(step_total)
        print('4', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        data_adv.detach().clone(), loss.detach().clone(), loss_total.detach().clone()
        del data_adv, loss, loss_total, data
        torch.cuda.empty_cache()
        print('5', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        model.zero_grad()
        end_time = time.time()


def calculate_loss(args, data, data_adv, model, optimizer, scheduler, step_total):
    if args.clean_weight > 0.:
        loss_clean = model(data)
    else:
        loss_clean = 0.
    print('3', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
    loss = model(data_adv)
    print(f'$$$$$$$$$$$$$$$$$$loss: {loss}, loss_clean: {loss_clean}*****************************')
    loss_total = args.clean_weight * loss_clean + (1 - args.clean_weight) * loss
    loss_total.backward()
    optimizer.step()
    optimizer.zero_grad()
    step_total += 1
    scheduler(step_total)
    print('4', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
    data_adv.detach().clone(), loss.detach().clone(), loss_total.detach().clone()
    del data_adv, loss, loss_total, data, model.input_ids, model.labels, model.attention_mask
    torch.cuda.empty_cache()
    print('5', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())


@torch.no_grad()
def compute_acc(logits, targets):
    preds_clean = logits.max(dim=1)[1].detach()
    acc = (preds_clean.eq(targets).sum() / targets.shape[0]).item() * 100
    return acc


def compute_loss(loss_str, embedding, targets, embedding_orig, logit_scale,
                 embedding_text_labels_norm=None, reduction='mean'):
    if loss_str == 'l2':
        loss = l2(out=embedding, targets=embedding_orig, reduction=reduction)
    elif loss_str == 'ce':
        loss = ce(
            out=embedding @ (logit_scale * embedding_text_labels_norm),
            targets=targets,
            reduction=reduction
        )
    else:
        raise ValueError(f'loss {loss_str} not supported')
    return loss


def l2(out, targets, reduction='none'):
    # squared l2 - it does not divide by the latent dimension
    # should have shape (batch_size, embedding_size)
    assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
    assert out.shape[0] > 1
    # Compute the element-wise squared error
    squared_error_batch = F.mse_loss(out, targets, reduction='none')
    if reduction == 'mean':
        squared_error_batch = torch.mean(squared_error_batch.sum(dim=1))
    else:
        squared_error_batch = squared_error_batch.sum(dim=1)
        assert squared_error_batch.shape == (out.shape[0],), f'{squared_error_batch.shape} != {(out.shape[0],)}'
    return squared_error_batch


def ce(out, targets, reduction='mean'):
    # out = logits
    assert out.shape[0] == targets.shape[0], (out.shape, targets.shape)
    assert out.shape[0] > 1

    return F.cross_entropy(out, targets, reduction=reduction)


if __name__ == '__main__':
    # set seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Parse command-line arguments
    args = parser.parse_args()  # parse_known_args()
    args.eps /= 255
    args.stepsize_adv /= 255
    # make sure there is no string in args that should be a bool
    assert not any([isinstance(x, str) and x in ['True', 'False'] for x in
                    args.__dict__.values()]), f'args contains a string that should be a bool: {args}'
    assert args.eval_freq % args.log_freq == 0, 'eval_freq must be a multiple of log_freq'

    if args.devices != '':
        # set cuda visible devices
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f'Number of GPUs available: {num_gpus}')
    else:
        print('No multiple GPUs available.')

    # set model name and output dir
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    args.finetuned_model_name = f'{args.clip_model_name}_{args.pretrained}_{args.dataset}_{args.loss}_{args.dataset}_{args.experiment_name}_{random_str}'
    args.finetuned_model_name = args.finetuned_model_name.replace('/', '_')
    args.output_dir = os.path.join(args.output_dir, args.finetuned_model_name)
    # run
    main(args, args)
