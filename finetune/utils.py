import logging
import os
import pprint
import socket

import numpy as np
import torch
import torch.distributed as dist


def add_initial_logs(config):
    logging.info(config.cmd)
    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(vars(config)))
    logging.info(f"PyTorch version: {torch.__version__}.")
    logging.info(f"NCCL Version {torch.cuda.nccl.version()}")
    logging.info(f"Hostname: {socket.gethostname()}.")
    logging.info(f"Using dataset: {config.dataset}")

    def init_distributed_mode(args):
        # launched with torch.distributed.launch
        try:
            import submitit
            job_env = submitit.JobEnvironment()
            logging.info(job_env)
            args.rank = int(job_env.global_rank)
            args.local_rank = int(job_env.local_rank)
            args.world_size = int(job_env.num_nodes) * args.ngpus

        except Exception as e:
            logging.info(e)
            logging.info('Will run the code on 8 GPUs.')
            args.rank, args.local_rank, args.world_size = 0, 0, torch.cuda.device_count()
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'

        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.cuda.set_device(args.local_rank)
        print('| distributed init (rank {}): {}'.format(
            args.rank, args.dist_url), flush=True)
        dist.barrier()
        setup_for_distributed(args.rank == 0)


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


def get_parameter_number(model):
    return np.sum([p.numel() for p in model.parameters() if p.requires_grad])


def setup_logging(config, rank=0):
    level = {'DEBUG': 10, 'ERROR': 40, 'FATAL': 50,
             'INFO': 20, 'WARN': 30
             }[config.logging_verbosity]
    format_ = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
    filename = '{}/log_{}_{}.logs'.format(config.train_dir, config.mode, rank)
    logging.basicConfig(filename=filename, level=level, format=format_, datefmt='%H:%M:%S')


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    try:
        import submitit
        job_env = submitit.JobEnvironment()
        logging.info(job_env)
        args.rank = int(job_env.global_rank)
        args.local_rank = int(job_env.local_rank)
        args.world_size = int(job_env.num_nodes) * args.ngpus

    except Exception as e:
        logging.info(e)
        logging.info('Will run the code on 8 GPUs.')
        args.rank, args.local_rank, args.world_size = 0, 0, torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.cuda.set_device(args.local_rank)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


class MessageBuilder:

    def __init__(self):
        self.msg = []

    def add(self, name, values, align=">", width=0, format=None):
        if name:
            metric_str = "{}: ".format(name)
        else:
            metric_str = ""
        values_str = []
        if type(values) != list:
            values = [values]
        for value in values:
            if format:
                values_str.append("{value:{align}{width}{format}}".format(
                    value=value, align=align, width=width, format=format))
            else:
                values_str.append("{value:{align}{width}}".format(
                    value=value, align=align, width=width))
        metric_str += '/'.join(values_str)
        self.msg.append(metric_str)

    def get_message(self):
        message = " | ".join(self.msg)
        self.clear()
        return message

    def clear(self):
        self.msg = []
