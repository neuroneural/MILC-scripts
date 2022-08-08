import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CheckSize(nn.Module):
    def forward(self, x):
        print("final size is", x.size())
        return x


class NatureCNN(nn.Module):
    def init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def __init__(self, input_channels, args):
        super().__init__()
        self.feature_size = args.feature_size
        # self.feature_size = 64
        self.hidden_size = self.feature_size
        self.downsample = not args.no_downsample
        self.input_channels = 1
        self.end_with_relu = args.end_with_relu
        self.args = args
        init_ = lambda m: self.init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        self.flatten = Flatten()

        if self.downsample:
            self.final_conv_size = 32 * 7 * 7
            self.final_conv_shape = (32, 7, 7)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                # nn.ReLU()
            )
        else:
            # self.final_conv_size = 64 * 41 * 8
            # self.final_conv_shape = (64, 41, 8)
            self.final_conv_size = 88 * 188
            self.final_conv_shape = (88,188)
            print("#####\n#####\n#####")
            print(self.feature_size)
            self.main = nn.Sequential(
                init_(nn.Conv2d(self.input_channels, 32, 4, stride=1)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=1)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 128, 4, stride=1)),
                nn.ReLU(),
                init_(nn.Conv2d(128, 64, 4, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                # nn.ReLU()
                # (64x16544 and 20992x256)
                # f7.shape = [64,88,188] -> 64x88*188 -> 64x16544
                # original final_conv_size = 64 * 41 * 8 = 20992
                # args.features = 256
            )
        self.train()

    def forward(self, inputs, fmaps=False):
        inputs = inputs.float() # added type cast
        f5 = self.main[:6](inputs)
        f7 = self.main[6:8](f5)
        # print(inputs.shape)
        # print(f5.shape)
        # print(f7.shape)
        # print(self.main[8])
        out = self.main[8:](f7)
        if self.end_with_relu:
            assert self.args.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)
        if fmaps:
            return {
                "f5": f5.permute(0, 2, 3, 1),
                "f7": f7.permute(0, 2, 3, 1),
                "out": out,
            }
        return out


class NatureOneCNN(nn.Module):
    def init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def __init__(self, input_channels, args):
        super().__init__()
        self.feature_size = args.feature_size
        self.hidden_size = self.feature_size
        self.downsample = not args.no_downsample
        self.input_channels = input_channels
        self.twoD = args.fMRI_twoD
        self.end_with_relu = args.end_with_relu
        self.args = args
        init_ = lambda m: self.init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        self.flatten = Flatten()

        if self.downsample:
            self.final_conv_size = 32 * 7 * 7
            self.final_conv_shape = (32, 7, 7)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                # nn.ReLU()
            )

        else:
            # problem with OASIS input is prolly here
            self.final_conv_size = 200 * 12
            self.final_conv_shape = (200, 12)
            print("feature_size = ", self.feature_size)
            self.main = nn.Sequential(
                init_(nn.Conv1d(input_channels, 64, 4, stride=1)),  # 0
                nn.ReLU(),
                init_(nn.Conv1d(64, 128, 4, stride=1)),
                nn.ReLU(),
                init_(nn.Conv1d(128, 200, 3, stride=1)),
                nn.ReLU(),
                Flatten(),  # 6
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                init_(nn.Conv1d(200, 128, 3, stride=1)),
                nn.ReLU(),
                # nn.ReLU()
            )
        # self.train()

    def forward(self, inputs, fmaps=False, five=False):
        # print("inputs = ", inputs.shape)
        # inputs =  torch.Size([7, 53, 20]) for COBRE
        # inputs =  torch.Size([6, 53, 26]) for OASIS
        f5 = self.main[:6](inputs)
        # print("f5 = ", f5.shape)
        # f5 =  torch.Size([7, 200, 12]) for COBRE
        # f5 =  torch.Size([6, 200, 18]) for OASIS
        out = self.main[6:8](f5)
        f5 = self.main[8:](f5)
        # print(inputs.shape)
        # print(f5.shape)
        # print(out.shape)

        if self.end_with_relu:
            assert self.args.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)
        if five:
            return f5.permute(0, 2, 1)
        if fmaps:
            return {
                "f5": f5.permute(0, 2, 1),
                # 'f7': f7.permute(0, 2, 1),
                "out": out,
            }
        return out
