import torch
import torch.nn as nn
from torch.nn import init

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None :
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None :
            nn.init.zeros_(m.bias)

class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        self.build(opt)
        print(self)

    def build(self, opt):
        nb_feat = opt.nb_feat_init_D
        
        block = [nn.Conv2d(opt.ch_inp + opt.ch_tar, nb_feat, kernel_size=4, stride=2, padding=1, bias=True),
                 nn.LeakyReLU(0.2)]

        for n in range(opt.nb_layer_D - 3):
            block += [nn.Conv2d(nb_feat, nb_feat*2, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(nb_feat*2), nn.LeakyReLU(0.2)]
            nb_feat *= 2

        block += [nn.Conv2d(nb_feat, nb_feat*2, kernel_size=4, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(nb_feat*2), nn.LeakyReLU(0.2)]
        nb_feat *= 2

        block += [nn.Conv2d(nb_feat, 1, kernel_size=4, stride=1, padding=1, bias=True), nn.Sigmoid()]

        self.model = nn.Sequential(*block)

    def forward(self, inp):
        return self.model(inp)


class BlockDown(nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out):
        super(BlockDown, self).__init__()
        self.build(nb_feat_in, nb_feat_out)

    def build(self, nb_feat_in, nb_feat_out):
        block = [nn.LeakyReLU(0.2),
                 nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(nb_feat_out)]
        self.model = nn.Sequential(*block)

    def forward(self, inp):
        return self.model(inp)

class BlockCenter(nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out):
        super(BlockCenter, self).__init__()
        self.build(nb_feat_in, nb_feat_out)

    def build(self, nb_feat_in, nb_feat_out):
        block = [nn.LeakyReLU(0.2),
                 nn.Conv2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=2, padding=1, bias=True),
                 nn.ReLU(),
                 nn.ConvTranspose2d(nb_feat_out, nb_feat_in, kernel_size=4, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(nb_feat_in),
                 nn.Dropout2d(0.5)]
        self.model = nn.Sequential(*block)

    def forward(self, inp):
        return self.model(inp)

class BlockUp(nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out, use_dropout):
        super(BlockUp, self).__init__()
        self.build(nb_feat_in, nb_feat_out, use_dropout)

    def build(self, nb_feat_in, nb_feat_out, use_dropout):
        block = [nn.ReLU(),
                 nn.ConvTranspose2d(nb_feat_in, nb_feat_out, kernel_size=4, stride=2, padding=1, bias=False),
                 nn.BatchNorm2d(nb_feat_out)]
        if use_dropout == True :
                block += [nn.Dropout2d(0.5)]
        self.model = nn.Sequential(*block)

    def forward(self, inp):
        return self.model(inp)


class UnetGenerator(nn.Module):
    def __init__(self, opt):
        super(UnetGenerator, self).__init__()
        self.opt = opt
        self.build()
        print(self)

    def build(self):
        nb_feat_after = self.opt.nb_feat_init_G
        self.block_down_0 = nn.Conv2d(self.opt.ch_inp, nb_feat_after, kernel_size=4, stride=2, padding=1, bias=True)
        self.block_up_0 = nn.Sequential(*[nn.ConvTranspose2d(nb_feat_after*2, self.opt.ch_tar, kernel_size=4, stride=2, padding=1, bias=True), nn.Tanh()])
        for i in range(self.opt.nb_down_G - 2):
            nb_feat_before = nb_feat_after
            nb_feat_after = min(nb_feat_after*2, 512)
            setattr(self, 'block_down_%d'%(i+1), BlockDown(nb_feat_before, nb_feat_after))
            use_dropout = True if i < 2 else False
            setattr(self, 'block_up_%d'%(i+1), BlockUp(nb_feat_after*2, nb_feat_before, use_dropout))
        nb_feat_before = nb_feat_after
        nb_feat_after = min(nb_feat_after*2, 512)
        self.block_center = BlockCenter(nb_feat_before, nb_feat_after)

    def forward(self, inp):

#        down_0 = self.block_down_0(inp)
#        down_1 = self.block_down_1(down_0)
#        down_2 = self.block_down_2(down_1)
#        down_3 = self.block_down_3(down_2)

#        center = self.block_center(down_3)

#        up_3 = self.block_up_3(torch.cat([center, down_3], 1))
#        up_2 = self.block_up_2(torch.cat([up_3, down_2], 1))
#        up_1 = self.block_up_1(torch.cat([up_2, down_1], 1))
#        up_0 = self.block_up_0(torch.cat([up_1, down_0], 1))

#        return up_0

        layers = [inp]
        for i in range(self.opt.nb_down_G-1):
            layers.append(getattr(self, 'block_down_%d'%(i)) (layers[-1]))

        last = self.block_center(layers[-1])

        for j in range(self.opt.nb_down_G -1):
            tmp = torch.cat([last, layers[-j-1]], 1)
            layer = getattr(self, 'block_up_%d'%(self.opt.nb_down_G - j - 2))
            last = layer(tmp)

        return last










if __name__ == '__main__' :
    from option import TrainOption
    opt = TrainOption().parse()
    network_D = PatchDiscriminator(opt)
    network_G = UnetGenerator(opt)

    inp = torch.ones((opt.batch_size, opt.ch_inp, opt.image_size, opt.image_size))
    tar = torch.ones((opt.batch_size, opt.ch_tar, opt.image_size, opt.image_size))

    print(inp.shape, tar.shape)

    output_D = network_D(torch.cat([inp, tar], 1))
    print(output_D.shape)

    output_G = network_G(inp)
    print(output_G.shape)


