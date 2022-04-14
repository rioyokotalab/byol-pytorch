import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOL
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# test model, a resnet 50

resnet = models.resnet50(pretrained=True)

# arguments

parser = argparse.ArgumentParser(description="byol-lightning-test")

parser.add_argument(
    "--image_folder",
    type=str,
    required=True,
    help='path to your folder of images for self-supervised learning')

parser.add_argument("--batch_size", type=int, default=256, help='bacth size')
parser.add_argument("--epochs", type=int, default=1000, help='epochs')
parser.add_argument("--num_gpus", type=int, default=2, help='num_gpus')
parser.add_argument("--lr", type=float, default=3e-4, help='lr')
parser.add_argument("--imaeg_size", type=int, default=256, help='image_size')

parser.add_argument("--result_path",
                    type=str,
                    required=True,
                    help='path to your folder of result')

args = parser.parse_args()

# constants

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
NUM_GPUS = args.num_gpus
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


# main

if __name__ == '__main__':
    train_ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
    # train_ds = ImagesDataset(args.image_folder, IMAGE_SIZE, "train")
    train_loader = DataLoader(train_ds,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
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
                                  moving_average_decay=0.99)

    # wandb.init(project="byol_pytorh_test",
    #            entity="tomo",
    #            name="pretrain-byol",
    #            config=args)
    logger = WandbLogger(project="byol_pytorh_test",
                         entity="tomo",
                         name="pretrain-byol",
                         log_model="all",
                         config=args)

    trainer = pl.Trainer(gpus=NUM_GPUS,
                         max_epochs=EPOCHS,
                         enable_checkpointing=True,
                         accumulate_grad_batches=1,
                         sync_batchnorm=True,
                         accelerator="gpu",
                         strategy="ddp",
                         logger=logger,
                         log_every_n_steps=1,
                         default_root_dir=args.result_path)

    trainer.fit(model, train_loader)
