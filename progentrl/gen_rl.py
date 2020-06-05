import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from math import pi, log
from .lp import LP
from .utils import save, load
import joblib
from collections import OrderedDict

from moses.metrics.utils import get_mol
import pytorch_lightning as pl

class oneDataSet(Dataset):
    def __init__(self):
        self.one_elem = [1]
        
    def __len__(self):
        return len(self.one_elem)
    
    def __getitem__(self, idx):
        return self.one_elem[0]


class GENTRL_RL(pl.LightningModule):
    '''
    GENTRL model
    '''
    def __init__(self,
                 reward_fn,
                 enc,
                 dec,
                 latent_descr,
                 feature_descr,
                 rl_batch_size = 200,
                 tt_int=40,
                 tt_type='usual', 
                 beta=0.01, 
                 gamma=0.1,
                 load_model=None
                ):
        super().__init__()
        
        self.reward_fn = reward_fn
        self.rl_batch_size = rl_batch_size
        
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
    
    
    def forward(self, num_samples):
        z = self.lp.sample(num_samples, 50 * ['s'] + ['m'])
        smiles = self.dec.sample(50, z, argmax=False)
        return smiles
    
    
    def training_step(self, batch, batch_idx):
        exploit_size = int(self.rl_batch_size * (1 - 0.3))
        exploit_z = self.lp.sample(exploit_size, 50 * ['s'] + ['m'])

        z_means = exploit_z.mean(dim=0)
        z_stds = exploit_z.std(dim=0)

        expl_size = int(self.rl_batch_size * 0.3)
        expl_z = torch.randn(expl_size, exploit_z.shape[1]).to(self.device)
        expl_z = 2 * expl_z * z_stds[None, :]
        expl_z += z_means[None, :]

        z = torch.cat([exploit_z, expl_z])
        smiles = self.dec.sample(50, z, argmax=False)
        zc = torch.zeros(z.shape[0], 1).to(z.device)
        conc_zy = torch.cat([z, zc], dim=1)
        log_probs = self.lp.log_prob(conc_zy, marg=50 * [False] + [True])
        log_probs += self.dec(smiles, z)
        r_list = [self.reward_fn(s) for s in smiles]

        rewards = torch.tensor(r_list).float().to(exploit_z.device)
        rewards_bl = rewards - rewards.mean()
        loss = -(log_probs * rewards_bl).mean()

        valid_sm = [s for s in smiles if get_mol(s) is not None]
        cur_stats = {
            'mean_reward': torch.tensor(sum(r_list) / len(smiles)),
            'valid_perc': torch.tensor(len(valid_sm) / len(smiles))
        }
        
        output_dict = OrderedDict({
            'loss': loss,
            'log': cur_stats,
            'progress_bar': cur_stats
        })
        
        return output_dict
    
    def configure_optimizers(self):
        lr_lp=1e-5
        lr_dec=1e-6
        
        optimizer = optim.Adam([
            {'params': self.lp.parameters()},
            {'params': self.dec.latent_fc.parameters(), 'lr': lr_dec}
        ], lr=lr_lp)
#         scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        return [optimizer]#, [scheduler]
    
    def train_dataloader(self):
        oneElementDataSet = oneDataSet()
        oneElementDataLoader = DataLoader(oneElementDataSet, batch_size=1)
        return oneElementDataLoader