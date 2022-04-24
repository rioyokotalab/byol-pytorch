import os
import sys
import time
import argparse
import multiprocessing
import logging
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import wandb
from pl_bolts.optimizers import LARS

from byol_pytorch import BYOL
from lr_schedule import CosineDecayLinearLRScheduler

_LR_PRESETS = {40: 0.45, 100: 0.45, 300: 0.3, 1000: 0.2}
_WD_PRESETS = {40: 1e-6, 100: 1e-6, 300: 1e-6, 1000: 1.5e-6}
_EMA_PRESETS = {40: 0.97, 100: 0.99, 300: 0.99, 1000: 0.996}


# mpi setup
def dist_setup(backend="nccl"):
    master_addr = os.getenv("MASTER_ADDR", default="localhost")
    master_port = os.getenv("MASTER_POST", default="8888")
    method = "tcp://{}:{}".format(master_addr, master_port)
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    if backend == "mpi":
        rank, world_size = -1, -1
    dist.init_process_group(backend=backend,
                            init_method=method,
                            rank=rank,
                            world_size=world_size)

    print("Rank: {}, Size: {}, Host: {}".format(dist.get_rank(),
                                                dist.get_world_size(),
                                                master_addr))


def dist_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def myget_rank():
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
    return rank


def myget_rank_size():
    rank = myget_rank()
    size = 1
    if dist.is_initialized():
        size = dist.get_world_size()
    return rank, size


# multi process print
def print_rank(*args):
    rank = myget_rank()
    print(f"rank: {rank}", *args)


def print0(*args):
    rank = myget_rank()
    if rank == 0:
        print(*args)


class SelfSupervisedLearner(nn.Module):

    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images, use_momentum=False):
        if use_momentum:
            self.on_before_zero_grad()
            return
        return self.learner(images)

    @torch.no_grad()
    def on_before_zero_grad(self):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

    def exclude_from_wt_decay(self, weight_decay, skip_list=("bias", "bn")):
        named_params = self.named_parameters()
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {
                "params": params,
                "weight_decay": weight_decay
            },
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]


# images dataset
def expand_greyscale(t):
    return t.expand(3, -1, -1)


class ImagesDataset(Dataset):
    IMAGE_EXTS = ['.jpg', '.png', '.jpeg']

    def __init__(self, folder, image_size, subset=None):
        super().__init__()
        self.folder = folder
        self.paths = []
        self.subset = subset
        ds_root = f"{folder}/{subset}" if subset else f"{folder}"

        for path in Path(ds_root).glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in self.IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train(model, train_loader, optimizer, lr_scheduler, epoch, all_epoch,
          device, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [batch_time, losses],
                             prefix='Train: ')

    len_loader = len(train_loader)
    initial_step = epoch * len_loader
    train_s_time = time.perf_counter()
    rank = myget_rank()
    model.train()

    for batch_idx, image in enumerate(train_loader):
        image = image.to(device)
        logger.info(f"batch_idx: {batch_idx}, after load")
        if rank == 0:
            logger.info(
                f"after load memory: {torch.cuda.memory_allocated(device)}")
        batch_s_time = time.perf_counter()

        loss = model(image)

        if rank == 0:
            logger.info(
                f"after forward memory: {torch.cuda.memory_allocated(device)}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model(image, use_momentum=True)
        lr_scheduler.step()

        if rank == 0:
            logger.info(
                f"after backward memory: {torch.cuda.memory_allocated(device)}"
            )

        batch_e_time = time.perf_counter()
        batch_exec_time = batch_e_time - batch_s_time
        batch_time.update(batch_exec_time)
        losses.update(loss.item(), image.size(0))
        global_step = initial_step + batch_idx

        head = f"epoch: {epoch}/{all_epoch} "
        head += f"batch idx: {global_step} {batch_idx}/{len_loader}"

        print_rank(head, f"loss: {loss}")
        print_rank(head, f"sec/batch: {batch_exec_time}s")
        logger.info(progress.display(batch_idx))

        if rank == 0:
            wandb.log({"iters/train/loss": loss}, commit=False)
            wandb.log({"iters/train/sec_batch": batch_exec_time}, commit=False)
            wandb.log({"iters/train/avg/loss": losses.avg}, commit=False)
            wandb.log({"iters": global_step})
    total_e_time = time.perf_counter()
    total_exec_time = total_e_time - train_s_time
    print_rank(epoch, "total_e_time:", total_exec_time, "s")
    return losses, progress


def main():
    # arguments
    parser = argparse.ArgumentParser(description="byol-lightning-test")

    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help='path to your folder of images for self-supervised learning')

    parser.add_argument("--batch_size",
                        type=int,
                        default=256,
                        help='bacth size')
    parser.add_argument("--epochs", type=int, default=1000, help='epochs')
    parser.add_argument("--num_gpus", type=int, default=2, help='num_gpus')
    # parser.add_argument("--lr", type=float, default=3e-4, help='lr')
    parser.add_argument("--imaeg_size",
                        type=int,
                        default=256,
                        help='image_size')
    parser.add_argument("--resnet_pretrain", action="store_true")

    parser.add_argument("--result_path",
                        type=str,
                        required=True,
                        help='path to your folder of result')
    parser.add_argument("--resume_path",
                        type=str,
                        default="",
                        help='path to your folder of resume')

    args = parser.parse_args()

    # constants
    global_batch_size = args.batch_size
    EPOCHS = args.epochs
    assert EPOCHS in _LR_PRESETS.keys()
    args.lr = _LR_PRESETS[EPOCHS]
    args.momentum = _EMA_PRESETS[EPOCHS]
    args.weight_decay = _WD_PRESETS[EPOCHS]
    # NUM_GPUS = args.num_gpus
    IMAGE_SIZE = args.imaeg_size
    NUM_WORKERS = multiprocessing.cpu_count()

    # test model, a resnet 50
    resnet = models.resnet50(pretrained=args.resnet_pretrain)

    train_ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
    # train_ds = ImagesDataset(args.image_folder, IMAGE_SIZE, "train")

    dist_setup()
    rank, world_size = myget_rank_size()
    is_cuda = torch.cuda.is_available()
    local_rank, node_num = 0, 1
    num_workers, ngpus = NUM_WORKERS // node_num, 1
    device = torch.device("cpu")
    if is_cuda:
        ngpus = torch.cuda.device_count()
        args.num_gpus = ngpus
        local_rank = rank % ngpus
        node_num = world_size // ngpus
        num_workers = num_workers // node_num
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    local_batch_size = global_batch_size // world_size

    train_sampler = DistributedSampler(train_ds,
                                       num_replicas=world_size,
                                       rank=rank)
    train_loader = DataLoader(dataset=train_ds,
                              batch_size=local_batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              sampler=train_sampler)

    model = SelfSupervisedLearner(resnet,
                                  image_size=IMAGE_SIZE,
                                  hidden_layer='avgpool',
                                  projection_size=256,
                                  projection_hidden_size=4096,
                                  moving_average_decay=args.momentum,
                                  use_momentum=True)

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=args.lr,
    #                              weight_decay=args.weight_decay)
    # optimizer = LARSWrapper(optimizer)
    params = model.exclude_from_wt_decay(weight_decay=args.weight_decay)
    optimizer = LARS(params,
                     lr=args.lr * world_size,
                     momentum=0.9,
                     weight_decay=args.weight_decay)
    global_steps = EPOCHS * len(train_loader)
    warmup_steps = 10 * len(train_loader)
    lr_scheduler = CosineDecayLinearLRScheduler(optimizer,
                                                global_batch_size,
                                                global_steps,
                                                warmup_steps,
                                                world_size,
                                                verbose=True)
    start_epoch = 0

    if args.resume_path != "":
        checkpoint = torch.load(args.resume_path, map_location="cpu")
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
        lr_scheduler.total_steps = global_steps

    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if rank == 0:
        wandb.init(project="byol_pytorh_test",
                   entity="tomo",
                   name="pretrain-byol",
                   config=args)

    ##########################################################################
    # log file prepare
    log_dir = os.path.join(args.result_path, "log_dir")
    os.makedirs(log_dir, exist_ok=True)
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if rank == 0:
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(plain_formatter)
        s_handler.setLevel(logging.DEBUG)
        logger.addHandler(s_handler)
    filename = f"console_gpu{rank}.log"
    f_handler = logging.FileHandler(os.path.join(log_dir, filename))
    f_handler.setFormatter(plain_formatter)
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)
    logger.propagate = False
    ##########################################################################

    resume_dir = os.path.join(args.result_path, "resume")
    os.makedirs(resume_dir, exist_ok=True)

    logger.info("start train")
    for epoch in range(start_epoch, EPOCHS):
        logger.info(f"start epoch {epoch}")
        losses, progress = train(model, train_loader, optimizer, lr_scheduler,
                                 epoch, EPOCHS, device, logger)

        if rank == 0:
            wandb.log({"epoch/train/loss/avg": losses.avg}, commit=False)
            wandb.log({"epoch/train/loss/base": losses.val}, commit=False)
            wandb.log({"epoch": epoch})
            state_dict = model.module.state_dict()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                },
                os.path.join(resume_dir,
                             "checkpoint_{:04d}.pth.tar".format(epoch)))


# main

if __name__ == '__main__':
    main()
