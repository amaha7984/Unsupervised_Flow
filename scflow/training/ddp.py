import torch
from torch.nn.parallel import DistributedDataParallel


def setup(backend="nccl"):
    torch.distributed.init_process_group(backend=backend, init_method="env://")


def unwrap_model(m):
    return m.module if isinstance(m, DistributedDataParallel) else m
