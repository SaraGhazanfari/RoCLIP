import argparse

from finetune.utils import str2bool

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
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
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

# distributed setting
parser.add_argument("--ncpus", type=int, default=4, help="Number of CPUS per GPU.")
parser.add_argument("--ngpus", type=int, default=4, help="Number of GPUs to use.")
parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes.")
parser.add_argument("--timeout", type=int, default=1440, help="Time of the Slurm job in minutes for training.")
parser.add_argument("--partition", type=str, default="nvidia", help="Partition to use for Slurm.")
parser.add_argument("--local", action='store_true', help="Execute with local machine instead of slurm.")
parser.add_argument('--devices', type=str, default='', help='Device IDs for CUDA')
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
    distributed training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
parser.add_argument("--constraint", default='h100', type=str)

# leftovers for model
parser.add_argument("--model_path", type=str)
parser.add_argument("--temperature", type=float)
parser.add_argument("--num_beams", type=int)
parser.add_argument("--precision", type=str)
parser.add_argument("--vision_encoder_pretrained", type=str)

# Logging parameters
parser.add_argument('--train_dir', type=str, default=None, help='Output directory')
parser.add_argument("--log_freq", type=int, default=1, help="The log is generated within this epoch frequency")
parser.add_argument("--eval_freq", type=int, default=10, help="The log is generated within this epoch frequency")
parser.add_argument("--logging_verbosity", type=str, default='INFO', help="Level of verbosity of the logs")
