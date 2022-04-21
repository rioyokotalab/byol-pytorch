import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from resnet import resnet50, resnet200


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


parser = argparse.ArgumentParser(description="Validate PyTorch weights")
parser.add_argument(
    "--wts",
    default="pretrain_res50x1.pth.tar",
    type=str,
    help="PyTorch weights to validate with imagenet validation set")
parser.add_argument("--valdir",
                    default="/datasets/imagenet/val",
                    type=str,
                    help="path to imagenet val directory")
parser.add_argument("--model",
                    default="resnet50",
                    type=str,
                    help="Model name. Valid: resnet50, resnet200")
parser.add_argument("--use_dist",
                    action="store_true",
                    help="use dist by torch option")


def main():
    args = parser.parse_args()

    if args.use_dist:
        dist_setup()
        rank, _ = myget_rank_size()
        ngpus = torch.cuda.device_count()
        local_rank = rank % ngpus
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    print_rank("start main")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    val_loader = DataLoader(
        datasets.ImageFolder(args.valdir, val_transform),
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    if args.model.lower() == "resnet200":
        model = resnet200(num_classes=1000).cuda()
    else:
        model = resnet50(num_classes=1000).cuda()

    state_dict = torch.load(args.wts)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    if args.use_dist:
        model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    print_rank("setup complete")

    validate(val_loader, model)


def validate(val_loader, model):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    model.eval()

    print_rank("start val")
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print_rank(progress.display(i))

        print_rank(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
            top1=top1, top5=top5))

    return top1.avg


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


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
