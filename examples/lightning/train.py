import os
# import sys
import argparse
import logging
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from byol_pytorch import BYOL
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ProgressBarBase

from utils import dist_setup, dist_cleanup
from utils import print_rank, myget_rank_size
from utils import myget_local_rank, myget_node_size

_LR_PRESETS = {40: 0.45, 100: 0.45, 300: 0.3, 1000: 0.2}
_WD_PRESETS = {40: 1e-6, 100: 1e-6, 300: 1e-6, 1000: 1.5e-6}
_EMA_PRESETS = {40: 0.97, 100: 0.99, 300: 0.99, 1000: 0.996}

# arguments

parser = argparse.ArgumentParser(description="byol-lightning-test")

parser.add_argument(
    "--image_folder",
    type=str,
    required=True,
    help='path to your folder of images for self-supervised learning')
parser.add_argument("--subset", type=str, default="", help='subset name')

parser.add_argument("--batch_size", type=int, default=256, help='bacth size')
parser.add_argument("--epochs", type=int, default=1000, help='epochs')
# parser.add_argument("--lr", type=float, default=3e-4, help='lr')
parser.add_argument("--imaeg_size", type=int, default=256, help='image_size')
parser.add_argument("--resnet_pretrain", action="store_true")
parser.add_argument("--accumulate_grad_batches", type=int, default=1)
parser.add_argument("--logging_set_detail",
                    action="store_true",
                    help="log detail for debug")

parser.add_argument("--result_path",
                    type=str,
                    required=True,
                    help='path to your folder of result')
parser.add_argument("--resume_path",
                    type=str,
                    default="",
                    help='path to your folder of resume')

args = parser.parse_args()

if args.logging_set_detail:
    logging.getLogger("pytorch_lightning").setLevel(pl._DETAIL)

dist_setup()

# test model, a resnet 50

resnet = models.resnet50(pretrained=args.resnet_pretrain)

# constants

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
args.lr = _LR_PRESETS[EPOCHS]
LR = args.lr
NUM_GPUS = 2
IMAGE_SIZE = args.imaeg_size
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()

# pytorch lightning module


class SelfSupervisedLearner(pl.LightningModule):

    def __init__(self, net, lr, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)
        self.lr = lr
        self.top1 = AverageMeter('Acc@1', ':6.2f')
        self.top5 = AverageMeter('Acc@5', ':6.2f')
        self.losses = AverageMeter('Loss', ':.4e')

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log("train/loss", loss)

        max_epochs = self.trainer.max_epochs
        max_steps = self.trainer.max_steps
        progress_callback = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, ProgressBarBase):
                progress_callback = callback
                break
        if progress_callback is not None:
            max_steps = progress_callback.main_progress_bar.total
        epoch = self.current_epoch
        global_step = self.global_step
        local_step = global_step
        if progress_callback is not None:
            local_step = global_step % max_steps
        head = f"Epoch: [{epoch}/{max_epochs}] "
        head += f"Iters: {global_step} [{local_step}/{max_steps}]"
        logger.info(f"{head} train/loss: {loss}")

        self.log("train/iters", global_step)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self.learner.online_encoder.net(images)
        loss = F.cross_entropy(output, target)
        epoch = self.current_epoch

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.losses.update(loss.item(), images.size(0))
        self.top1.update(acc1[0], images.size(0))
        self.top5.update(acc5[0], images.size(0))

        self.log(f"val/loss/{epoch}", loss)
        self.log(f"val/acc1/{epoch}", acc1)
        self.log(f"val/acc5/{epoch}", acc5)
        logger.info(f"epoch: {epoch} {batch_idx} val/loss: {loss}")
        return {"loss": loss.item(), "acc1": acc1, "acc5": acc5}

    def validation_epoch_end(self, outputs) -> None:
        print_rank(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
            top1=self.top1, top5=self.top5))
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
            top1=self.top1, top5=self.top5))
        epoch = self.current_epoch
        self.log("epoch/val/loss", self.losses.avg)
        self.log("epoch/val/acc1", self.top1.avg)
        self.log("epoch/val/acc5", self.top5.avg)
        logger.info(f"epoch: {epoch} end val/loss: {self.losses.avg}")
        self.top1.reset()
        self.top5.reset()
        self.losses.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()


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
    """
    Computes the accuracy over the k top predictions
    for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# images dataset


def expand_greyscale(t):
    return t.expand(3, -1, -1)


class ImagesDataset(Dataset):

    def __init__(self, folder, image_size, subset=""):
        super().__init__()
        self.folder = folder
        self.paths = []
        self.subset = subset
        use_subset = subset != "" or subset is not None
        ds_root = f"{folder}/{subset}" if use_subset else f"{folder}"

        for path in Path(ds_root).glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print_rank(f'{len(self.paths)} images found')
        # print(f'{len(self.paths)} images found')

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


# main

if __name__ == '__main__':
    rank, world_size = myget_rank_size()
    is_cuda = torch.cuda.is_available()
    local_rank, node_num = myget_local_rank(), myget_node_size()
    num_workers, ngpus = NUM_WORKERS // node_num, NUM_GPUS
    if is_cuda:
        ngpus = torch.cuda.device_count()
        args.num_gpus = ngpus
        local_rank = rank % ngpus
        node_num = world_size // ngpus
        num_workers = num_workers // node_num
        # torch.cuda.set_device(local_rank)
        # device = torch.device("cuda", local_rank)
    num_workers = 8

    ##########################################################################
    # log file prepare
    log_dir = os.path.join(args.result_path, "log_dir")
    os.makedirs(log_dir, exist_ok=True)
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # if rank == 0:
    #     s_handler = logging.StreamHandler(stream=sys.stdout)
    #     s_handler.setFormatter(plain_formatter)
    #     s_handler.setLevel(logging.DEBUG)
    #     logger.addHandler(s_handler)
    filename = f"console_gpu{rank}.log"
    f_handler = logging.FileHandler(os.path.join(log_dir, filename))
    f_handler.setFormatter(plain_formatter)
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)
    logger.propagate = False
    ##########################################################################

    local_batch_size = BATCH_SIZE // world_size
    acc_grad_num = args.accumulate_grad_batches
    # local_batch_size = local_batch_size // acc_grad_num

    print_rank("start main")
    logger.info("start main")
    # print_rank("start main num_workers:", num_workers)
    # train_ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
    # train_ds = ImagesDataset(args.image_folder, IMAGE_SIZE, "train")
    # subset_prefix = "ILSVRC2012_img_"
    # subset = subset_prefix + "train"
    subset = args.subset
    train_ds = ImagesDataset(args.image_folder, IMAGE_SIZE, subset)
    train_loader = DataLoader(train_ds,
                              batch_size=local_batch_size,
                              num_workers=num_workers,
                              shuffle=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    valdir = os.path.join(args.image_folder, "val")
    val_loader = DataLoader(
        datasets.ImageFolder(valdir, val_transform),
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    moving_average_decay = _EMA_PRESETS[EPOCHS]
    model = SelfSupervisedLearner(resnet,
                                  lr=args.lr * world_size,
                                  image_size=IMAGE_SIZE,
                                  hidden_layer='avgpool',
                                  projection_size=256,
                                  projection_hidden_size=4096,
                                  moving_average_decay=moving_average_decay,
                                  use_momentum=True)

    print_rank("start setup train")
    logger.info("start setup train")

    csv_dir = os.path.join(args.result_path, "csv_dir")
    csv_logger = CSVLogger(save_dir=csv_dir, flush_logs_every_n_steps=1)
    tf_dir = os.path.join(args.result_path, "tf_logs")
    tf_logger = TensorBoardLogger(save_dir=tf_dir)
    lightning_loggers = [csv_logger, tf_logger]

    if rank == 0:
        # wandb.init(project="byol_pytorh_test",
        #            entity="tomo",
        #            name="pretrain-byol",
        #            config=args)
        wandb_logger = WandbLogger(project="byol_pytorh_test",
                                   entity="tomo",
                                   name="pretrain-byol",
                                   log_model=False,
                                   config=args)
        lightning_loggers.append(wandb_logger)

    checkpoint_root_path = os.path.join(args.result_path, "checkpoints_root")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor="step",
        mode="max",
        # every_n_train_steps=1,
        dirpath=checkpoint_root_path,
        every_n_epochs=1,
        filename="byol-{epoch:04d}-{step}",
        verbose=False)

    trainer = pl.Trainer(
        gpus=ngpus,
        num_nodes=node_num,
        # num_processes=ngpus,
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        accumulate_grad_batches=acc_grad_num,
        # sync_batchnorm=True,
        sync_batchnorm=False,
        accelerator="gpu",
        strategy="ddp",
        logger=lightning_loggers,
        log_every_n_steps=1,
        default_root_dir=args.result_path)

    print_rank("start train")
    logger.info("start train")

    trainer.fit(model,
                train_loader,
                val_dataloaders=val_loader,
                ckpt_path=args.resume_path)

    dist_cleanup()
