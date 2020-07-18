import os
import torch
import torch.nn as nn
import torch.optim as optim
from math import pi, log
from .lp import LP
from .utils import save, load
import joblib
from collections import OrderedDict

from moses.metrics.utils import get_mol
import pytorch_lightning as pl
    

class GENTRL_VAELP(pl.LightningModule):
    '''
    GENTRL model
    '''
    def __init__(self,
                 enc,
                 dec,
                 train_loader,
                 latent_descr,
                 feature_descr,
                 tt_int=40,
                 tt_type='usual', 
                 beta=0.01, 
                 gamma=0.1,
                 load_model=None
                ):
        super().__init__()
        
        self.train_loader = train_loader
        self.num_latent = len(latent_descr)
        self.num_features = len(feature_descr)

        self.latent_descr = latent_descr
        self.feature_descr = feature_descr

        self.tt_int = tt_int
        self.tt_type = tt_type

        self.enc = enc
        self.dec = dec
        
        self.beta = beta
        self.gamma = gamma
        
        self.lp = LP(distr_descr=self.latent_descr + self.feature_descr,
                     tt_int=self.tt_int, tt_type=self.tt_type)
        
        if load_model is not None:
            self = load(self, load_model)
        
        #Epoch Variables
        self.buf = None

    def get_elbo(self, x, y):
        means, log_stds = torch.split(self.enc(x),
                                      len(self.latent_descr), dim=1)
        latvar_samples = (means + torch.randn_like(log_stds) *
                          torch.exp(0.5 * log_stds))

        rec_part = self.dec(x, latvar_samples).mean()

        normal_distr_hentropies = (log(2 * pi) + 1 + log_stds).sum(dim=1)

        latent_dim = len(self.latent_descr)
        condition_dim = len(self.feature_descr)

        zy = torch.cat([latvar_samples, y], dim=1)
        log_p_zy = self.lp.log_prob(zy)

        y_to_marg = latent_dim * [True] + condition_dim * [False]
        log_p_y = self.lp.log_prob(zy, marg=y_to_marg)

        z_to_marg = latent_dim * [False] + condition_dim * [True]
        log_p_z = self.lp.log_prob(zy, marg=z_to_marg)
        log_p_z_by_y = log_p_zy - log_p_y
        log_p_y_by_z = log_p_zy - log_p_z

        kldiv_part = (-normal_distr_hentropies - log_p_zy).mean()

        elbo = rec_part - self.beta * kldiv_part
        elbo = elbo + self.gamma * log_p_y_by_z.mean()

        return elbo, {
            'elbo_loss': -elbo,
            'rec': rec_part,
            'kl': kldiv_part,
            'log_p_y_by_z': log_p_y_by_z.mean(),
            'log_p_z_by_y': log_p_z_by_y.mean()
        }
    
    def reinit_from_data(self):
        to_reinit = True
        for x_batch, y_batch in self.train_dataloader():
            y_batch = y_batch.float().to(self.lp.tt_cores[0].device)
            if to_reinit:
                if len(y_batch.shape) == 1:
                    y_batch = y_batch.view(-1, 1).contiguous()
                    
                if (self.buf is None) or (self.buf.shape[0] < 5000):
                    enc_out = self.enc(x_batch)
                    means, log_stds = torch.split(enc_out,
                                                  len(self.latent_descr),
                                                  dim=1)
                    z_batch = (means + torch.randn_like(log_stds) *
                               torch.exp(0.5 * log_stds))
                    cur_batch = torch.cat([z_batch, y_batch], dim=1)
                    if self.buf is None:
                        self.buf = cur_batch
                    else:
                        self.buf = torch.cat([self.buf, cur_batch])
                else:
                    descr = len(self.latent_descr) * [0]
                    descr += len(self.feature_descr) * [1]
                    self.lp.reinit_from_data(self.buf, descr)
                    self.lp.to(self.device)
                    self.buf = None
                    to_reinit = False
            else:
                print('reinit Done from data')
                break
    
    
    def forward(self, num_samples):
        z = self.lp.sample(num_samples, 50 * ['s'] + ['m'])
        smiles = self.dec.sample(50, z, argmax=False)
        return smiles
    
    
    def training_step(self, batch, batch_idx):
        if self.current_epoch in [0, 1, 5] and batch_idx==0:
            self.reinit_from_data()
        
        x_batch, y_batch = batch
        elbo, cur_stats = self.get_elbo(x_batch, y_batch)
        loss = -elbo

        output_dict = OrderedDict({
            'loss': loss,
            'log': cur_stats,
            'progress_bar': cur_stats
        })

        return output_dict
    
    def configure_optimizers(self):
        lr=1e-3
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
#         scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        return [optimizer]#, [scheduler]
    
    def train_dataloader(self):
        return self.train_loader