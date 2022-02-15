

import warnings
warnings.filterwarnings('ignore')

from option import TestOption
opt = TestOption().parse()

import torch
import torch.nn as nn
import random
import numpy as np

import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
ngpu = torch.cuda.device_count()
cuda = ngpu > 0
device = torch.device('cuda' if cuda else 'cpu')
print(ngpu, device)


path_model = '%s/model'%(opt.root_save)


from imageio import imsave
from pipeline import CustomDataset, SnapMaker
from torch.utils import data
import matplotlib.pyplot as plt

snap_maker = SnapMaker()
dataset = CustomDataset(opt)
dataloader = data.DataLoader(dataset=dataset, batch_size=opt.batch_size,
                                shuffle=opt.is_train, num_workers=opt.num_workers)

nb_data = len(dataset)
nb_batch = len(dataloader)
print(nb_data, nb_batch)

from network import UnetGenerator

network_G = UnetGenerator(opt)
state = torch.load('%s/%04d.pt'%(path_model, opt.epoch_target))
network_G.load_state_dict(state['network_G'])

if ngpu > 1 :
    network_G = nn.DataParallel(network_G)
network_G = network_G.to(device)
network_G.eval()

snaps = []

for idx, (inp, tar) in enumerate(dataloader):

        inp = inp.to(device)
        tar = tar

        gen = network_G(inp)
        gen = gen.detach().cpu().numpy()

        tar = np.transpose(tar, (0,2,3,1))
        gen = np.transpose(gen, (0,2,3,1))

        tar = np.hstack([tar[n] for n in range(4)])
        gen = np.hstack([gen[n] for n in range(4)])

        snap = np.vstack([tar, gen])
        snap = snap_maker(snap)
        snaps.append(snap)

plt.imshow(snaps[0])
plt.show()
