import argparse

class BaseOption():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--prefix', type=str, default='denoising')
        self.parser.add_argument('--seed', type=int, default=1220)
        self.parser.add_argument('--image_size', type=int, default=256)
        self.parser.add_argument('--ch_inp', type=int, default=1)
        self.parser.add_argument('--ch_tar', type=int, default=3)

        self.parser.add_argument('--nb_layer_D', type=int, default=5)
        self.parser.add_argument('--nb_feat_init_D', type=int, default=64)

        self.parser.add_argument('--nb_down_G', type=int, default=8)
        self.parser.add_argument('--nb_feat_init_G', type=int, default=64)


        self.parser.add_argument('--root_data', type=str,
                                 default='./dataset')
        self.parser.add_argument('--root_save', type=str,
                                 default='./result')

    def parse(self):
        return self.parser.parse_args()

class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument('--is_train', type=bool, default=True)
        self.parser.add_argument('--gpu_ids', type=str, default='')
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--eps', type=float, default=1e-8)
        self.parser.add_argument('--weight_L1_loss', type=float, default=100.)

        self.parser.add_argument('--epoch_total', type=int, default=10)
        self.parser.add_argument('--report_frequency', type=int, default=10)



class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()

        self.parser.add_argument('--is_train', type=bool, default=False)
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')
        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--num_workers', type=int, default=16)
        self.parser.add_argument('--epoch_test', type=int, default=100)