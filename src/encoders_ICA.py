import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CheckSize(nn.Module):
    def forward(self, x):
        print("final size is", x.size())
        return x

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResNet_13(nn.Module):
    def __init__(self, block, layers, num_classes, args):
        self.inplanes = 64
        super(ResNet_13, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)
        self.feature_size = args.feature_size
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool3d(3)
        self.fc = nn.Linear(9216 * block.expansion, 512)
        self.fc2 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        # x = block(self.inplanes, planes, stride=stride, downsample=downsample)
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)        
        x = self.layer1(x)        
        x = self.layer2(x)        
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc(x)
        x = self.fc2(x1)
        return [x,x1]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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
            # print("#####\n#####\n#####")
            # print(self.feature_size)
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
