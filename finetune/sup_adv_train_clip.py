import logging
import sys
import time
import uuid
from datetime import datetime
from os.path import realpath
from pathlib import Path

import submitit
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from finetune import utils
from train.datasets import COCOFlickrDataset
from train.pgd_train import pgd
from vlm_eval.attacks.apgd import apgd
from vlm_eval.utils import force_cudnn_initialization, get_eval_model

sys.path.append("open_flamingo")
import os
import string
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from training.scheduler import cosine_lr
import wandb
from train.utils import init_wandb
from open_flamingo.eval.models.utils import unwrap_model
from train.utils import str2bool

import argparse


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    shared_folder = '/home/sg7457'
    if not os.path.exists(shared_folder):
        shared_folder = '/home/sara'
    os.makedirs(str(shared_folder), exist_ok=True)
    init_file = Path(shared_folder) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def set_config(args):
    # process arguments
    os.makedirs('./trained_models', exist_ok=True)
    path = realpath('./trained_models')
    if args.train_dir is None:  # and config.local_rank == 0:
        args.start_new_model = True
        folder = datetime.now().strftime("%Y-%m-%d_%H.%M.%S_%f")[:-2]
        args.train_dir = f'{path}/{folder}'
        os.makedirs(args.train_dir)
        args.ckpt = f'{args.train_dir}/checkpoints'
        os.makedirs(args.aug_ckpt)
        os.makedirs(args.head_ckpt)
    else:
        args.train_dir = f'{path}/{args.train_dir}'
        args.ckpt = f'{args.train_dir}/checkpoints'
    return args


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
parser.add_argument('--mode', type=str, default='')

parser.add_argument('--log_freq', type=int, default=1, help='Logging frequency')
parser.add_argument('--eval_freq', type=int, default=50, help='Evaluation frequency')
parser.add_argument('--train_dir', type=str, default=None, help='Output directory')
parser.add_argument('--save_checkpoints', type=str2bool, default=True, help='Save 10 training checkpoints')

# distributed setting
parser.add_argument("--ngpus", type=int, default=4, help="Number of GPUs to use.")
parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes.")
parser.add_argument("--timeout", type=int, default=1440, help="Time of the Slurm job in minutes for training.")
parser.add_argument("--partition", type=str, default="gpu_p13", help="Partition to use for Slurm.")
parser.add_argument("--local", action='store_true', help="Execute with local machine instead of slurm.")
parser.add_argument('--devices', type=str, default='', help='Device IDs for CUDA')
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
    distributed training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")

# leftovers for model
parser.add_argument("--model_path", type=str)
parser.add_argument("--temperature", type=float)
parser.add_argument("--num_beams", type=int)
parser.add_argument("--precision", type=str)
parser.add_argument("--vision_encoder_pretrained", type=str)


class LLaVAFinetune:
    def __init__(self, args):
        self.args = args
        # setup wandb

    def __call__(self, *args, **kwargs):
        utils.setup_logging(self.args)
        utils.add_initial_logs(self.args)

        utils.init_distributed_mode(self.args)
        self.message = utils.MessageBuilder()

        if self.args.wandb:
            init_wandb(
                project_name='clip-finetune',
                model_name=self.args.finetuned_model_name,
                config=vars(self.args)
            )
        else:
            wandb.init(mode='disabled')

        if self.args.optimizer_state != '':
            assert self.args.start_step > 0
            assert str(self.args.start_step) in self.args.optimizer_state
            assert self.args.pretrained in ['', 'none']
            self.args.pretrained = self.args.optimizer_state.replace('_opt', '')

        image_dir_path = f'{self.args.imagenet_root}/train2014'
        annotations_path = f'{self.args.imagenet_root}/annotations/captions_train2014.json'

        model = get_eval_model(self.args, self.args.__dict__, adversarial="none")  # TinyLLAVA(args, main_device)  #

        self._get_data(annotations_path, image_dir_path, model)
        params = self._prepare_model(model)

        self._get_optimizer(params)

        # set scheduler
        self.scheduler = cosine_lr(self.optimizer, self.args.lr, self.args.warmup, self.args.steps)

        # compute amount of epochs
        total_epochs = self.args.steps / len(self.dataloader)
        print(f'train for {total_epochs} epochs')
        self.args.total_epochs = total_epochs

        # finetune
        self.step_total = self.args.start_step
        epoch = 0
        self.num_steps = 0
        if self.args.local_rank == 0:
            logging.info('Beginning training from epoch: 1')
        while self.step_total < self.args.steps:
            if self.sampler:
                self.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            print(f'Epoch {epoch} done.')
            epoch += 1
            return

        # save final model
        torch.save(unwrap_model(self.model).get_vision_tower().vision_tower.state_dict(),
                   f'{self.args.ckpt}/final.pt')
        torch.save(self.optimizer.state_dict(), f'{self.args.ckpt}/final_opt.pt')

        if self.args.train_dir.endswith('_temp'):
            os.rename(self.args.train_dir, self.args.train_dir[:-5])

    def _get_optimizer(self, params):
        if self.args.opt == 'adamw':
            self.optimizer = torch.optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.wd)
        elif self.args.opt == 'sgd':
            self.optimizer = torch.optim.SGD(
                params,
                lr=self.args.lr,
                momentum=self.args.momentum_sgd,
                weight_decay=self.args.wd
            )
        else:
            raise ValueError(f'Optimizer {self.args.optimizer} not supported.')
        if self.args.optimizer_state != '':
            self.optimizer.load_state_dict(torch.load(self.args.optimizer_state))

    def _prepare_model(self, model):
        force_cudnn_initialization()
        device_id = 0
        model.set_device(device_id)
        params = model.model.get_vision_tower().vision_tower.parameters()
        if args.ngpus > 1 and args.nnodes > 1:
            self.model = DistributedDataParallel(model.model, device_ids=[self.args.local_rank])
        elif args.ngpus > 1:
            self.model = DataParallel(model.model, device_ids=range(args.ngpus))
        else:
            self.model = model.model
        return params

    def _get_data(self, annotations_path, image_dir_path, model):
        dataset = COCOFlickrDataset(model=model,
                                    image_processor=model._prepare_images,
                                    image_dir_path=image_dir_path,
                                    annotations_path=annotations_path,
                                    transform=None,
                                    prefix='COCO_train2014_'
                                    )
        self.sampler = None
        if self.args.ngpus > 1 and self.args.nnodes > 1:
            self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        self.dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=8,
                                     drop_last=True, sampler=self.sampler)

    def train_one_epoch(self, epoch):
        unwrap_model(self.model).get_vision_tower().vision_model.train()
        start_time, end_time = time.time(), time.time()

        for idx, (data, input_ids, labels, attention_mask) in enumerate(self.dataloader):
            logging.info(f'{idx}/{len(self.dataloader)} Time:{round(end_time - start_time, 4)}')
            start_time = time.time()
            data, input_ids, labels, attention_mask = data.to('cuda:0'), input_ids.to('cuda:0'), labels.to(
                'cuda:0'), attention_mask.to('cuda:0')

            if args.attack == 'pgd':
                data_adv = pgd(
                    forward=self.model,
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
                    model=self.model,
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
                    self.model(pixel_values=data, input_ids=input_ids, attention_mask=attention_mask,
                               past_key_values=None,
                               inputs_embeds=None, labels=labels).loss)
            else:
                loss_clean = 0.
            # print('3', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            out = self.model(pixel_values=data_adv, input_ids=input_ids, attention_mask=attention_mask,
                             past_key_values=None,
                             inputs_embeds=None, labels=labels)
            loss = torch.mean(out.loss)
            loss_total = args.clean_weight * loss_clean + (1 - args.clean_weight) * loss
            loss_total.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step_total += 1
            self.scheduler(self.step_total)
            data_adv.detach().clone(), loss.detach().clone(), loss_total.detach().clone()
            del data_adv, data
            self.model.zero_grad()
            self.num_steps += 1
            if idx % self.args.log_freq == self.args.log_freq - 1 and self.args.local_rank == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.message.add("epoch", epoch + idx / len(self.dataloader), format="4.2f")
                self.message.add("lr", lr, format=".6f")
                self.message.add("num_steps", self.num_steps, format="1d")
                self.message.add("train loss", loss, format=".4f")
                self.message.add("train total loss", loss_total, format=".4f")
                self.message.add("time", int(time.time() - start_time) / 60, format=".2f")
                logging.info(self.message.get_message())


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
    args.finetuned_model_name = f'{args.clip_model_name}_{args.pretrained}_{args.dataset}_{args.loss}_{args.dataset}_{args.mode}'
    args.finetuned_model_name = args.finetuned_model_name.replace('/', '_')
    ncpus = 40
    # default: set tasks_per_node equal to number of gpus
    tasks_per_node = args.ngpus
    cluster = 'slurm' if not args.local else 'local'
    executor = submitit.AutoExecutor(folder=args.train_dir, cluster=cluster)
    # if not config.local:
    executor.update_parameters(
        gpus_per_node=args.ngpus,
        nodes=args.nnodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=ncpus // tasks_per_node,
        stderr_to_stdout=True,
        slurm_job_name=f'{args.train_dir[-4:]}_{args.mode}',
        slurm_partition=args.partition,
        slurm_signal_delay_s=0,
        slurm_mem='64GB',
        timeout_min=args.timeout,
    )
    args.dist_url = get_init_file().as_uri()
    args.cmd = f"python3 {' '.join(sys.argv)}"
    set_config(args)
    finetune = LLaVAFinetune(args)
    job = executor.submit(finetune)
    job_id = job.job_id
    folder = args.train_dir.split('/')[-1]
    print(f"Submitted batch job {job_id} in folder {folder}")
