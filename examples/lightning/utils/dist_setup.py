import os

import torch.distributed as dist

import pytorch_lightning.plugins.environments.lightning_environment as le
from pytorch_lightning.utilities.distributed import init_dist_connection


def dist_setup(backend="nccl"):
    master_addr = os.getenv("MASTER_ADDR", default="localhost")
    master_port = os.getenv("MASTER_POST", default="8888")
    method = "tcp://{}:{}".format(master_addr, master_port)
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))

    # setting for pytorch lightning
    local_rank = int(os.getenv("OMPI_COMM_WORLD_NODE_RANK", "0"))
    local_size = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", "1"))
    node_rank = (rank - local_rank) // local_size
    node_num = world_size // local_size
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["NODE_RANK"] = str(node_rank)
    os.environ["NODE_NUM"] = str(node_num)

    le_environment = le.LightningEnvironment()

    init_dist_connection(cluster_environment=le_environment,
                         torch_distributed_backend=backend,
                         global_rank=rank,
                         world_size=world_size,
                         init_method=method)

    str_print = f"Rank: {dist.get_rank()}, Size: {dist.get_world_size()}, "
    str_print += f"Host: {master_addr}, Local Rank: {local_rank}, "
    str_print += f"NODE_RANK: {node_rank}, Local Size: {local_size}, "
    str_print += f"Node_num: {node_num}"
    print(str_print)


# for pytorch_lightning variable
# get node num
def myget_node_size():
    return int(os.getenv("NODE_NUM", "1"))


# get node id
def myget_node_rank():
    return int(os.getenv("NODE_RANK", "0"))


# get local rank on node
def myget_local_rank():
    return int(os.getenv("LOCAL_RANK", "0"))


def dist_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def myget_rank_size():
    rank, world_size = 0, 1
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return rank, world_size


def myget_rank():
    rank, _ = myget_rank_size()
    return rank


# multi process print
def print0(*args):
    rank = myget_rank()
    if rank == 0:
        print(*args)


def print_rank(*args):
    rank, world_size = myget_rank_size()
    digit = len(str(world_size))
    str_rank = str(rank).zfill(digit)
    print(f"rank: {str_rank}", *args)
