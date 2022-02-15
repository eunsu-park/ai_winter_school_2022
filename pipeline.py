import torch
from torch.utils import data
import numpy as np
from imageio import imread
from glob import glob

class SnapMaker:
    def __call__(self, data):
        tmp = (data + 1.) * (255./2.)
        return np.clip(tmp, 0, 255).astype(np.uint8)

class CustomDataset(data.Dataset):
    def __init__(self, opt):
        self.is_train = opt.is_train
        if self.is_train == True :
            pattern = '%s/train/*.JPEG'%(opt.root_data)
        else :
            pattern = '%s/test/*.JPEG'%(opt.root_data)
        self.list_data = glob(pattern)
        self.nb_data = len(self.list_data)
        self.image_size = opt.image_size

    def __len__(self):
        return self.nb_data

    def __getitem__(self, idx):
        tar = imread(self.list_data[idx]).astype(np.float32)
        tar = np.transpose(tar, (2,0,1))
        if self.is_train == True :
            tar = self.cut_train(tar)
        else :
            tar = self.cut_test(tar)
        inp = self.rgb2gray(tar)
        inp = self.norm(inp)
        tar = self.norm(tar)
        inp = torch.from_numpy(inp)
        tar = torch.from_numpy(tar)
        return inp, tar

    def rgb2gray(self, img):
        gray = 0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]
        return gray

    def norm(self, img):
        return img * (2./255.) - 1.

    def cut_train(self, img):
        c, h, w = img.shape
        h_init = np.random.randint(0, h-self.image_size)
        w_init = np.random.randint(0, w-self.image_size)
        img_cut = img[:, h_init:h_init+self.image_size, w_init:w_init+self.image_size]
        return img_cut

    def cut_test(self, img):
        c, h, w = img.shape
        h_factor = h // self.image_size
        w_factor = w // self.image_size

        h_new = h_factor * self.image_size
        w_new = w_factor * self.image_size

        h_init = np.random.randint(0, h - h_new)
        w_init = np.random.randint(0, w - w_new)

        img_cut = img[:, h_init:h_init+h_new, w_init:w_init+w_new]
        return img_cut



if __name__ == '__main__' :
    from option import TrainOption
    opt = TrainOption().parse()

    dataset = CustomDataset(opt)
    nb_data = len(dataset)
    print(nb_data)

    inp, tar = dataset[0]
    print(inp.shape, tar.shape)

    dataloader = data.DataLoader(dataset=dataset, batch_size=opt.batch_size,
                                    shuffle=opt.is_train, num_workers=opt.num_workers)
    nb_batch = len(dataloader)
    print(nb_batch)

    epochs = 1
    for epoch in range(epochs):
        for idx, (inp, tar) in enumerate(dataloader):
            print(epoch, idx, inp.shape, tar.shape, inp.numpy().min(), inp.numpy().max())

    print(inp.shape)
    print(tar.shape)
    inp = inp[0]
    tar = tar[0]
    inp = np.transpose(inp, (1,2,0))[:,:,0]
    tar = np.transpose(tar, (1,2,0))
    print(inp.shape)
    print(tar.shape)

    from imageio import imsave
    imsave('inp.png', inp)
    imsave('tar.png', tar)


