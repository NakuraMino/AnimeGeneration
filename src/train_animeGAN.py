from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import WandbLogger 
import dataloader 
import trainer 
import argparse

def train_animeGAN(root_dir, anime_dir, photo_dir, logger_name, ckpt=None, 
                   steps=100000, use_gpu=False):
    wandb_logger = WandbLogger(name=logger_name, project="animeGAN")
    pl_trainer = Trainer(gpus=int(use_gpu),default_root_dir=root_dir, max_steps=steps,
                      resume_from_checkpoint=ckpt, logger=wandb_logger, 
                      check_val_every_n_epoch=2, track_grad_norm=2)
    train_dl = dataloader.getPhotoAndAnimeDataloader(anime_dir, photo_dir)
    val_dl = dataloader.getPhotoAndAnimeDataloader(anime_dir, photo_dir)
    animeGAN = trainer.AnimeGANTrainer()
    pl_trainer.fit(animeGAN, train_dl=train_dl, val_dl=val_dl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an AnimeGAN model')
    parser.add_argument('-n', '--name', type=str, help='name of the model experiment', required=True)
    parser.add_argument('-s', '--steps', type=int, default=100000, help='max number of steps')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('-rd', '--root_dir', type=str, default="/home/nakuram/CSEP573-NeRF/experiments/", help='directory to save models')
    parser.add_argument('-l', '--ckpt', type=str, default=None, help='load/resume from checkpoint. Should be a path (e.g. ./checkpoints/last.ckpt')
    parser.add_argument('-p', '--photo_dir', type=str, default=None, help='directory with photo dataset')
    parser.add_argument('-a', '--anime_dir', type=str, default=None, help='directory with anime dataset')

    args = parser.parse_args()

    train_animeGAN(args.root_dir, args.anime_dir, args.photo_dir, args.name, args.ckpt, args.steps, args.gpu)
    