import os
import argparse
import logging
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOL
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from utils import dist_setup, dist_cleanup
from utils import print_rank, myget_rank_size
from utils import myget_local_rank, myget_node_size

# arguments

parser = argparse.ArgumentParser(description="byol-lightning-test")

parser.add_argument(
    "--image_folder",
    type=str,
    required=True,
    help='path to your folder of images for self-supervised learning')

parser.add_argument("--batch_size", type=int, default=256, help='bacth size')
parser.add_argument("--epochs", type=int, default=1000, help='epochs')
parser.add_argument("--lr", type=float, default=3e-4, help='lr')
parser.add_argument("--imaeg_size", type=int, default=256, help='image_size')
parser.add_argument("--resnet_pretrain", action="store_true")
parser.add_argument("--logging_set_detail",
                    action="store_true",
                    help="log detail for debug")

parser.add_argument("--result_path",
                    type=str,
                    required=True,
                    help='path to your folder of result')

args = parser.parse_args()

if args.logging_set_detail:
    logging.getLogger("pytorch_lightning").setLevel(pl._DETAIL)

dist_setup()

# test model, a resnet 50

resnet = models.resnet50(pretrained=args.resnet_pretrain)

# constants

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
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

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log("train/loss", loss)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()


# images dataset


def expand_greyscale(t):
    return t.expand(3, -1, -1)


class ImagesDataset(Dataset):

    def __init__(self, folder, image_size, subset=None):
        super().__init__()
        self.folder = folder
        self.paths = []
        self.subset = subset
        ds_root = f"{folder}/{subset}" if subset else f"{folder}"

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

    print_rank("start main")
    # print_rank("start main num_workers:", num_workers)
    # train_ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
    train_ds = ImagesDataset(args.image_folder, IMAGE_SIZE, "train")
    train_loader = DataLoader(train_ds,
                              batch_size=BATCH_SIZE,
                              num_workers=num_workers,
                              shuffle=True)
    # val_ds = ImagesDataset(args.image_folder, IMAGE_SIZE, "val")
    # val_loader = DataLoader(val_ds,
    #                         batch_size=BATCH_SIZE,
    #                         num_workers=NUM_WORKERS,
    #                         shuffle=False)

    model = SelfSupervisedLearner(resnet,
                                  lr=args.lr,
                                  image_size=IMAGE_SIZE,
                                  hidden_layer='avgpool',
                                  projection_size=256,
                                  projection_hidden_size=4096,
                                  moving_average_decay=0.99,
                                  use_momentum=True)

    # wandb.init(project="byol_pytorh_test",
    #            entity="tomo",
    #            name="pretrain-byol",
    #            config=args)
    logger = WandbLogger(project="byol_pytorh_test",
                         entity="tomo",
                         name="pretrain-byol",
                         log_model="all",
                         config=args)

    print_rank("start setup train")
    trainer = pl.Trainer(
        gpus=ngpus,
        num_nodes=node_num,
        # num_processes=ngpus,
        max_epochs=EPOCHS,
        enable_checkpointing=True,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        accelerator="gpu",
        strategy="ddp",
        logger=logger,
        log_every_n_steps=1,
        default_root_dir=args.result_path)
    print_rank("start train")

    trainer.fit(model, train_loader)

    dist_cleanup()
