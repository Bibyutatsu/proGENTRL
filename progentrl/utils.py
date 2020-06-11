import torch
import os
import joblib
from .lp import LP

def save(model, folder_to_save='./'):
    if not os.path.isdir(folder_to_save):
        os.makedirs(folder_to_save)
    torch.save(model.enc.state_dict(), os.path.join(folder_to_save, 'enc.model'))
    torch.save(model.dec.state_dict(), os.path.join(folder_to_save, 'dec.model'))
    torch.save(model.lp.state_dict(), os.path.join(folder_to_save, 'lp.model'))

    joblib.dump(model.lp.order, os.path.join(folder_to_save, 'order.pkl'))
    return True


def load(model, folder_to_load='./'):
    enc_file = os.path.join(folder_to_load, 'enc.model')
    dec_file = os.path.join(folder_to_load, 'dec.model')
    lp_file = os.path.join(folder_to_load, 'lp.model')
    lp_order = os.path.join(folder_to_load, 'order.pkl')

    if not os.path.isdir(folder_to_load):
        print("Cannot Load the Models: Folder not Found ", folder_to_load)
        return None
    if not os.path.isfile(enc_file):
        print("Cannot Load the Models: Enc Model not Found ")
        return None

    if not os.path.isfile(dec_file):
        print("Cannot Load the Models: Dec Model not Found ")
        return None

    if not os.path.isfile(lp_file):
        print("Cannot Load the Models: LP Model not Found ")
        return None

    if not os.path.isfile(lp_order):
        print("Cannot Load the Models: LP Order not Found ")
        return None

    order = joblib.load(lp_order)
    model.lp = LP(distr_descr=model.latent_descr + model.feature_descr,
                 tt_int=model.tt_int, tt_type=model.tt_type,
                 order=order)

    model.enc.load_state_dict(torch.load(enc_file))
    model.dec.load_state_dict(torch.load(dec_file))
    model.lp.load_state_dict(torch.load(lp_file))
    return model

def free_cuda_mem(tensors):
    for tensor in tensors:
        del tensor
    torch.cuda.empty_cache()