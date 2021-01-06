import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from networks import AdaINGen, MsImageDis, segmentor
from utils.utils import weights_init, get_model_list, get_scheduler
from utils.dice_loss import DiceLoss
import config_file as config

class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters, opts):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        self.opts = opts

        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.seg = segmentor(num_classes=2, channels=hyperparameters['input_dim_b'], hyperpars=hyperparameters['seg'])

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.seg_opt = torch.optim.SGD(self.seg.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters['lr_policy'], hyperparameters=hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters['lr_policy'], hyperparameters=hyperparameters)
        self.seg_scheduler = get_scheduler(self.seg_opt, 'constant', hyperparameters=None)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        self.criterion_seg = DiceLoss(ignore_index=hyperparameters['seg']['ignore_index'])

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters, target_a, iters):
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        if iters >= hyperparameters['guide_gen_iters']:
            config.task = 0
            self.seg.eval()
            self.pred_x_ab = self.seg(x_ab)
            self.seg.train()

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # semantic loss ab
        if iters >= hyperparameters['guide_gen_iters']:
            self.loss_sem_ab, _ = self.criterion_seg(self.pred_x_ab, target_a)
        else:
            self.loss_sem_ab = 0

        # only use semantic loss when segmentor has reasonably low loss
        if not hasattr(self, 'loss_seg_ab') or self.loss_seg_ab.detach().item() > -0.3:
            self.loss_sem_ab = 0

        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['sem_w'] * self.loss_sem_ab

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def seg_update(self, x_a,  x_b, target_a, target_b):
        self.seg.train()
        self.seg_opt.zero_grad()
        s_b = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        with torch.no_grad():
            # encode
            c_a, _ = self.gen_a.encode(x_a)
            # decode (cross domain)
            x_ab = self.gen_b.decode(c_a, s_b)

        config.task = 0
        self.pred_x_ab = self.seg(x_ab.detach())

        config.task = 1
        self.pred_x_b = self.seg(x_b)

        self.loss_seg_ab, _ = self.criterion_seg(self.pred_x_ab, target_a)
        self.loss_seg_b, _ = self.criterion_seg(self.pred_x_b, target_b)

        self.loss_seg_total = self.loss_seg_ab + self.loss_seg_b
        self.loss_seg_total.backward()
        self.seg_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.seg_scheduler is not None:
            self.seg_scheduler.step()

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_aba, x_bab, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], [], [], []
        for i in range(x_b.size(0)):
            # encode
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            # decode (within domain)
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            # decode (cross domain)
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
            # encode again
            c_b_recon, s_a_recon = self.gen_a.encode(x_ba1[-1])
            c_a_recon, s_b_recon = self.gen_b.encode(x_ab1[-1])
            x_aba.append(self.gen_a.decode(c_a_recon, s_a_fake))
            x_bab.append(self.gen_b.decode(c_b_recon, s_b_fake))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        x_aba, x_bab = torch.cat(x_aba), torch.cat(x_bab)

        self.train()
        return x_a, x_a_recon, x_aba, x_ab1, x_ab2, x_b, x_b_recon, x_bab, x_ba1, x_ba2

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, 'gen')
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, 'dis')
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load segmentor
        last_model_name = get_model_list(checkpoint_dir, 'seg')
        state_dict = torch.load(last_model_name)
        self.seg.load_state_dict(state_dict)

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'opt.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        state_dict = torch.load(os.path.join(checkpoint_dir, 'opt_seg.pt'))
        self.seg_opt.load_state_dict(state_dict)

        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters['lr_policy'], hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters['lr_policy'], hyperparameters, iterations)
        self.seg_scheduler = get_scheduler(self.seg_opt, 'constant', None, iterations)

        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        seg_name = os.path.join(snapshot_dir, 'seg_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'opt.pt')
        opt_seg_name = os.path.join(snapshot_dir, 'opt_seg.pt')

        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save(self.seg.state_dict(), seg_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
        torch.save(self.seg_opt.state_dict(), opt_seg_name)
