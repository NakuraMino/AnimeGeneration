import torch 
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from discriminator import *
from losses import *
from generator import *
import utilities as utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AnimeGANTrainer(LightningModule):

    def __init__(self, lambda_content=1.5, lambda_gray=3, lambda_color=10, lambda_adv=300):
        super(AnimeGANTrainer, self).__init__()
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

        # GAN training hack
        self.epoch = -1

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: 
            self.epoch += 1
        photos = batch['photo']
        anime = batch['anime']
        images = batch['image']
        labels = batch['label']

        g_opt, d_opt = self.optimizers()

        if self.epoch % 5 != 4:
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

            # manual backpropagation
            g_opt.zero_grad()
            self.manual_backward(loss)
            g_opt.step()

            return loss
        else:
            # train discriminator
            pred_labels = self.discriminator(images)
            loss = self.adversarial_loss(labels, pred_labels)

            self.log('train/discriminator_loss', loss, batch_size=4)
            
            # manual backpropagation 
            d_opt.zero_grad()
            self.manual_backward(loss)
            d_opt.step()

            return loss

    def validation_step(self, batch, batch_idx):
        photos = batch['photo']
        anime = batch['anime']
        images = batch['image']
        labels = batch['label']

        # train generator
        gen_images = self.generator(photos)
        pred_labels = self.discriminator(gen_images)
        fake_labels = torch.ones(pred_labels.shape, device=device)

        con_loss = self.content_loss(gen_images, photos)
        gray_loss = self.grayscale_loss(gen_images, anime)
        color_loss = self.color_recon_loss(gen_images, photos)
        adv_loss = self.adversarial_loss(pred_labels, fake_labels)

        loss = self.l_con * con_loss + self.l_gray * gray_loss + self.l_color * color_loss + self.l_adv * adv_loss

        self.log('val/generator_loss', loss, batch_size=4)
        self.log('val/con_loss', con_loss, batch_size=4)
        self.log('val/gray_loss', gray_loss, batch_size=4)
        self.log('val/color_loss', color_loss, batch_size=4)
        self.log('val/adv_loss', adv_loss, batch_size=4)
        
        # train discriminator
        pred_labels = self.discriminator(images)
        loss = self.adversarial_loss(labels, pred_labels)

        self.log('val/discriminator_loss', loss, batch_size=4)

        if batch_idx == 0:
            gen_images = utils.torch_to_numpy(gen_images, is_standardized_image=True)
            N, H, W, C = gen_images.shape
            self.logger.log_image(key='generated', images=[gen_images[i] for i in range(N)], caption=[f'val/{i}.png' for i in range(N)])
        return loss


    def configure_optimizers(self):
        gen_optim = torch.optim.Adam(self.generator.parameters(), lr=0.00008)
        dis_optim = torch.optim.Adam(self.discriminator.parameters(), lr=0.00016)
        return [gen_optim, dis_optim], []