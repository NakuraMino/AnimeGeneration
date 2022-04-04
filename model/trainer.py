import torch 
import torch.nn as nn
import torch.nn.functional as F
from pytotrch_lightning import LightningModule

from discriminator import *
from losses import *
from generator import *
import utilities as utils
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AnimeGANTrainer(LightningModule):

    def __init__(self, lambda_content=0.01, lambda_gray=10, lambda_color=5, lambda_adv=100):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.VGG = getVGGConv4_4()

        # loss weights
        self.l_con = lambda_content
        self.l_gray = lambda_gray
        self.l_color = lambda_color
        self.l_adv = lambda_adv

        # losses
        self.content_loss = ContentLoss(self.VGG)
        self.grayscale_loss = GrayscaleStyleLoss(self.VGG)
        self.color_recon_loss = ColorReconLoss()
        self.adversarial_loss = nn.MSELoss()

    def training_step(self, batch, batch_idx, optimizer_idx):
        photos = batch['photos']
        anime = batch['anime']

        if optimizer_idx == 0:
            # train generator
            gen_images = self.generator(photos)
            pred_labels = self.discriminator(gen_images)
            fake_labels = torch.ones(pred_labels.shape, device=device)

            con_loss = self.content_loss(gen_images, photos)
            gray_loss = self.grayscale_loss(gen_images, anime)
            color_loss = self.color_recon_loss(gen_images, photos)
            adv_loss = self.adversarial_loss(pred_labels, fake_labels)

            loss = self.l_con * con_loss + self.l_gray * gray_loss + self.l_color * color_loss + self.l_adv * adv_loss
            self.log('train/generator_loss', loss, batch_size=4)
            self.log('train/con_loss', con_loss, batch_size=4)
            self.log('train/gray_loss', gray_loss, batch_size=4)
            self.log('train/color_loss', color_loss, batch_size=4)
            self.log('train/adv_loss', adv_loss, batch_size=4)

            return loss
        elif optimizer_idx == 1:
            # train discriminator
            gen_images = self.generator(photos)
            pred_gen_labels = self.discriminator(gen_images)
            gen_labels = torch.zeros(pred_gen_labels.shape, device=device)
            gen_adv_loss = self.adversarial_loss(gen_labels, pred_gen_labels)

            pred_anime_labels = self.discriminator(anime)
            anime_labels = torch.ones(pred_anime_labels.shape, device=device)
            anime_adv_loss = self.adversarial_loss(anime_labels, pred_anime_labels)

            loss = gen_adv_loss + anime_adv_loss

            self.log('train/discriminator_loss', loss, batch_size=4)
            self.log('train/gen_adv_loss', gen_adv_loss, batch_size=4)
            self.log('train/anime_adv_loss', anime_adv_loss, batch_size=4)
            return loss


    def configure_optimizers(self):
        gen_optim = torch.optim.Adam(self.generator.parameters(), lr=0.00008)
        dis_optim = torch.optim.Adam(self.discriminator.parameters(), lr=0.00016)
        return [gen_optim, dis_optim], []