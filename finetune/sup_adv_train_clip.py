import copy
import logging
import sys
import time
import uuid
from datetime import datetime
from os.path import realpath
from pathlib import Path

import submitit
from torch.nn.parallel import DistributedDataParallel

from finetune import utils
from finetune.datasets import COCOFlickrDataset
from finetune.param import parser
from finetune.pgd_train import pgd
from finetune.utils import init_wandb
from llava.constants import IGNORE_INDEX
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

from open_flamingo.eval.models.utils import unwrap_model


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
        os.makedirs(args.ckpt)
    else:
        args.train_dir = f'{path}/{args.train_dir}'
        args.ckpt = f'{args.train_dir}/checkpoints'
    return args


class LLaVAFinetune:
    def __init__(self, args):
        self.args = args
        # setup wandb

    def __call__(self, *args, **kwargs):
        utils.setup_logging(self.args)
        utils.add_initial_logs(self.args)

        model = get_eval_model(self.args, self.args.__dict__, adversarial="none")
        self.vision_teacher = copy.deepcopy(model.model.get_vision_tower().vision_tower)
        self.normalizer = model.normalizer
        self.tokenizer = model.tokenizer
        logging.info('Model loaded successfully.')
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

        # TinyLLAVA(args, main_device)  #

        self._get_data(model)
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
        self._save_model(self.args.steps)

        if self.args.train_dir.endswith('_temp'):
            os.rename(self.args.train_dir, self.args.train_dir[:-5])

    def _save_model(self, step_num):
        torch.save(unwrap_model(self.model).get_vision_tower().vision_tower.state_dict(),
                   f'{self.args.ckpt}/final_{step_num}.pt')
        torch.save(self.optimizer.state_dict(), f'{self.args.ckpt}/final_opt_{step_num}.pt')

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
        logging.info('Optimizer loaded successfully :)')

    def _prepare_model(self, model):
        force_cudnn_initialization()
        params = model.model.get_vision_tower().vision_tower.parameters()
        if args.ngpus > 1:  # and args.nnodes > 1:
            self.model = DistributedDataParallel(model.model.cuda(), device_ids=[self.args.local_rank],
                                                 find_unused_parameters=True)
            logging.info('model loaded successfully on a multiple gpus and nodes!')
        # elif args.ngpus > 1:
        #     self.model = DataParallel(model.model.cuda(), device_ids=range(args.ngpus))
        #     logging.info('model loaded successfully on a multiple gpus and one node!')
        else:
            self.model = model.model.cuda()
            logging.info('model loaded successfully on a single gpu and node!')
        return params

    def _get_data(self, model):
        image_dir_path = f'{self.args.imagenet_root}/train2017'
        annotations_path = f'{self.args.imagenet_root}/annotations/captions_train2017.json'
        dataset = COCOFlickrDataset(model=model,
                                    image_processor=model._prepare_images,
                                    image_dir_path=image_dir_path,
                                    annotations_path=annotations_path,
                                    transform=None,
                                    prefix=''
                                    # prefix='COCO_train2014_'
                                    )
        image_dir_path = f'{self.args.imagenet_root}/val2017'
        annotations_path = f'{self.args.imagenet_root}/annotations/captions_val2017.json'
        val_dataset = COCOFlickrDataset(model=model,
                                        image_processor=model._prepare_images,
                                        image_dir_path=image_dir_path,
                                        annotations_path=annotations_path,
                                        transform=None,
                                        prefix=''
                                        # prefix='COCO_train2014_'
                                        )
        self.sampler = None
        if self.args.ngpus > 1:
            self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        self.dataloader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=8,
                                     drop_last=True, sampler=self.sampler)
        self.valloader = DataLoader(val_dataset, batch_size=5, num_workers=8,
                                    drop_last=True)
        logging.info('Trainloader created successfully!')

    def train_one_epoch(self, epoch):
        unwrap_model(self.model).get_vision_tower().vision_tower.train()
        start_time = time.time()
        for idx, (data, input_ids, labels, attention_mask) in enumerate(self.dataloader):
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
                    verbose=False,
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
                    self.model(images=data, input_ids=input_ids, attention_mask=attention_mask,
                               past_key_values=None,
                               inputs_embeds=None, labels=labels).loss.sum())
            else:
                loss_clean = 0.
            # print('3', torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            vision_embedding = list()
            with torch.no_grad():
                teacher_vision_embedding = self.vision_teacher(self.normalizer(data))

            def hook(module, input, output):
                vision_embedding.append(output)

            hook_handle = self.model.get_vision_tower().vision_tower.register_forward_hook(hook)

            out = self.model(images=self.normalizer(data_adv), input_ids=input_ids, attention_mask=attention_mask,
                             past_key_values=None, inputs_embeds=None, labels=labels)
            loss = out.loss.sum()

            vision_loss = torch.nn.MSELoss()(teacher_vision_embedding, vision_embedding[0])
            loss_total = args.clean_weight * loss_clean + (1 - args.clean_weight) * loss + vision_loss
            loss_total.backward()
            self.optimizer.step()
            hook_handle.remove()
            self.optimizer.zero_grad()
            self.step_total += 1
            self.scheduler(self.step_total)
            data_adv.detach().clone(), loss.detach().clone(), vision_loss.detach().clone(), loss_total.detach().clone()
            del data_adv, data
            self.model.zero_grad()
            if idx % self.args.log_freq == self.args.log_freq - 1 and self.args.local_rank == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.message.add("epoch", epoch + idx / len(self.dataloader), format="4.2f")
                self.message.add("lr", lr, format=".10f")
                self.message.add("num_steps", self.step_total, format="1d")
                self.message.add("total", self.args.steps, format="1d")
                self.message.add("Adv loss", loss, format=".4f")
                self.message.add("clean loss", loss_clean, format=".4f")
                self.message.add("Vision loss", vision_loss, format=".4f")
                self.message.add("time", int(time.time() - start_time) / 60, format=".2f")
                logging.info(self.message.get_message())
                start_time = time.time()

            if idx % self.args.eval_freq == self.args.eval_freq - 1 and self.args.local_rank == 0:
                self.evaluate()

            if idx % 2000 == 1999:
                self._save_model(idx + 1)

    def evaluate(self):
        for idx, (data, input_ids, labels, attention_mask) in enumerate(self.valloader):
            data, input_ids, labels, attention_mask = data.to('cuda:0'), input_ids.to('cuda:0'), labels.to(
                'cuda:0'), attention_mask.to('cuda:0')
            out = unwrap_model(self.model).generate(images=data, input_ids=input_ids, attention_mask=attention_mask,
                                                    past_key_values=None, min_new_tokens=0, max_new_tokens=20,
                                                    num_beams=3,
                                                    length_penalty=-2.0, labels=labels)

            for batch_idx in range(labels.shape[0]):
                gt = labels[batch_idx]
                gt = [t.item() for t in gt if t != IGNORE_INDEX]
                gt = self.tokenizer.decode(gt).replace('</s>', '').replace('<unk>', '')
                logging.info(f'gt: {gt}')
                out[batch_idx, out[batch_idx] == -200] = 1
                pred = self.tokenizer.decode(out[batch_idx], skip_special_tokens=True).split('ASSISTANT: ')[-1]
                logging.info(f'pred:{pred}')

            if idx == 1:
                break


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
    args = set_config(args)

    tasks_per_node = args.ngpus
    cluster = 'slurm' if not args.local else 'local'
    executor = submitit.AutoExecutor(folder=args.train_dir, cluster=cluster)
    # if not config.local:
    executor.update_parameters(
        gpus_per_node=args.ngpus,
        # gres=f'gpu:{args.ngpus}',
        constraint=args.constraint,
        nodes=args.nnodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=args.ncpus,
        stderr_to_stdout=True,
        slurm_job_name=f'{args.train_dir[-4:]}_{args.mode}',
        # slurm_partition=args.partition,
        slurm_signal_delay_s=0,
        slurm_mem='64GB',
        timeout_min=args.timeout,
        additional_parameters={'mail_type': 'BEGIN',
                               'mail_user': 'sg7457@nyu.edu'}
    )
    args.dist_url = get_init_file().as_uri()
    args.cmd = f"python3 {' '.join(sys.argv)}"
    finetune = LLaVAFinetune(args)
    job = executor.submit(finetune)
    job_id = job.job_id
    folder = args.train_dir.split('/')[-1]
    print(f"Submitted batch job {job_id} in folder {folder}")
