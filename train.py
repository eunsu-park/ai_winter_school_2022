if __name__ == '__main__' :

    import warnings
    warnings.filterwarnings('ignore')

    from option import TrainOption
    opt = TrainOption().parse()

    import torch
    import torch.nn as nn
    import random
    import numpy as np

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchmark = False

    import os, time
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    ngpu = torch.cuda.device_count()
    cuda = ngpu > 0
    device = torch.device('cuda' if cuda else 'cpu')
    print(ngpu, device)


    path_model = '%s/model'%(opt.root_save)
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    path_snap = '%s/snap'%(opt.root_save)
    if not os.path.exists(path_snap):
        os.makedirs(path_snap)

    from imageio import imsave
    from pipeline import CustomDataset, SnapMaker
    from torch.utils import data

    snap_maker = SnapMaker()
    dataset = CustomDataset(opt)
    dataloader = data.DataLoader(dataset=dataset, batch_size=opt.batch_size,
                                    shuffle=opt.is_train, num_workers=opt.num_workers)
    nb_data = len(dataset)
    nb_batch = len(dataloader)
    print(nb_data, nb_batch)

    from network import PatchDiscriminator, UnetGenerator, weights_init

    network_D = PatchDiscriminator(opt).apply(weights_init)
    network_G = UnetGenerator(opt).apply(weights_init)

    if ngpu > 1 :
        network_D = nn.DataParallel(network_D)
        network_G = nn.DataParallel(network_G)

    network_D = network_D.to(device)
    network_G = network_G.to(device)
    network_D.train()
    network_G.train()

    optim_D = torch.optim.Adam(network_D.parameters(), lr = opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    optim_G = torch.optim.Adam(network_G.parameters(), lr = opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)

    criterion = nn.BCELoss().to(device)
    L1criterion = nn.L1Loss().to(device)


    palette = '[%d/%d][%d] Loss_D: %5.3f Loss_G: %5.3f Loss_L: %5.3f Time: %dsec'
    epochs = 0
    iterations = 0
    t0 = time.time()
    losses_D = []
    losses_G = []
    losses_L = []

    while epochs < opt.epoch_total :

        for idx, (inp, tar) in enumerate(dataloader):
            inp = inp.to(device)
            tar = tar.to(device)

            optim_D.zero_grad()
            gen = network_G(inp)

            output_D_real = network_D(torch.cat([inp, tar], 1))
            target_D_real = torch.ones_like(output_D_real, dtype=torch.float).to(device)

            output_D_fake = network_D(torch.cat([inp, gen], 1))
            target_D_fake = torch.zeros_like(output_D_fake, dtype=torch.float).to(device)

            loss_D_real = criterion(output_D_real, target_D_real)
            loss_D_fake = criterion(output_D_fake, target_D_fake)
            loss_D = (loss_D_real + loss_D_fake)/2.

            loss_D.backward()
            optim_D.step()

            optim_G.zero_grad()
            gen = network_G(inp)

            output_G_fake = network_D(torch.cat([inp, gen], 1))
            target_G_fake = torch.ones_like(output_G_fake, dtype=torch.float).to(device)

            loss_G_fake = criterion(output_G_fake, target_G_fake)
            loss_L = L1criterion(tar, gen) * opt.weight_L1_loss
            loss_G = loss_G_fake + loss_L

            loss_G.backward()
            optim_G.step()

            losses_D.append(loss_D.item())
            losses_G.append(loss_G_fake.item())
            losses_L.append(loss_L.item())

            iterations += 1

            if iterations % opt.report_frequency == 0 :

                loss_D_mean = np.mean(losses_D)
                loss_G_mean = np.mean(losses_G)
                loss_L_mean = np.mean(losses_L)

                report = (epochs, opt.epoch_total, iterations,
                    loss_D_mean, loss_G_mean, loss_L_mean, time.time()-t0)
                print(palette % report)

                t0 = time.time()
                losses_D = []
                losses_G = []
                losses_L = []

                network_G.eval()

                snap_inp = inp[:4]
                snap_tar = tar[:4]
                snap_gen = network_G(snap_inp)

                snap_tar = snap_tar.detach().cpu().numpy()
                snap_gen = snap_gen.detach().cpu().numpy()

                snap_tar = np.transpose(snap_tar, (0,2,3,1))
                snap_gen = np.transpose(snap_gen, (0,2,3,1))

                snap_tar = np.hstack([snap_tar[n] for n in range(4)])
                snap_gen = np.hstack([snap_gen[n] for n in range(4)])

                snap = np.vstack([snap_tar, snap_gen])
                snap = snap_maker(snap)
                imsave('%s/%04d.%07d.png'%(path_snap, epochs, iterations), snap)

                network_G.train()

        epochs += 1


        if cuda and ngpu > 1 :
            state = {'network_D':network_D.module.state_dict(),
                    'network_G':network_G.module.state_dict(),
                    'optimizer_D':optim_D.state_dict(),
                    'optimizer_G':optim_G.state_dict()}
        else :
            state = {'network_D':network_D.state_dict(),
                    'network_G':network_G.state_dict(),
                    'optimizer_D':optim_D.state_dict(),
                    'optimizer_G':optim_G.state_dict()}

        torch.save(state, '%s/%04d.pt'%(path_model, epochs))
