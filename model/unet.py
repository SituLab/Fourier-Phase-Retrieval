import torch
import torch.nn as nn
import model

activation = 'relu'
if activation == 'prelu':
    activation = (nn.PReLU())
elif activation == 'leakyrelu':
    activation = (nn.LeakyReLU(0.2))
elif activation == 'tanh':
    activation = (nn.Tanh())
elif activation == 'relu':
    activation = (nn.ReLU())
elif activation == 'elu':
    activation = (nn.ELU())

class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            activation,
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            activation
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            activation
        )

    def forward(self, x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out = self.Conv_BN_ReLU_2(x)
        out_2 = self.downsample(out)
        return out, out_2


class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            activation,
            nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            activation
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_ch),
            activation
        )

    def forward(self, x, out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        cat_out = torch.cat((x_out, out), dim=1)
        return cat_out


class UNet(nn.Module):
    def __init__(self, dim_x=128, dim_y=128, batchsize=8,in_channels = 1, output_channels = 1):
        super(UNet, self).__init__()
        out_channels = [2 ** (i) * 32 for i in range(5)]  # [32, 64, 128, 256, 512] [96, 192, , , ]
        # self.fc = nn.Sequential(nn.Linear(in_features=dim_x*dim_y, out_features=dim_x*dim_y, bias=True),activation)
        # 下采样
        self.d1 = DownsampleLayer(in_channels, out_channels[0])  # 1-32......
        self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 32-64
        self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 64-128
        self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 128-256
        # 上采样
        self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 256-256
        self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 512-128
        self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 256-64
        self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 128-32
        # 输出
        self.o = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            activation,
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            activation,
            nn.Conv2d(out_channels[0], output_channels, 3, 1, 1),
            nn.Sigmoid(),# BCELoss  nn.Sigmoid()
            )
        self.batchsize = batchsize
        self.dim_x = dim_x
        self.dim_y = dim_y
        # self.P1 = nn.Parameter(torch.randn(1, 1, dim_x, dim_y))


    def forward(self, x):
        # x1 = self.fc(x.view(self.batchsize, -1))
        out_1, out1 = self.d1(x)#1.view(self.batchsize, 1, self.dim_x, self.dim_y))
        out_2, out2 = self.d2(out1)
        out_3, out3 = self.d3(out2)
        out_4, out4 = self.d4(out3)
        out5 = self.u1(out4, out_4)
        out6 = self.u2(out5, out_3)
        out7 = self.u3(out6, out_2)
        out8 = self.u4(out7, out_1)
        out = self.o(out8)
        return out


if __name__ == '__main__':
    test_data = torch.randn(5, 1, 512, 512).cuda()
    net = UNet(512, 512, 1, output_channels = 2).cuda()
    # b = net(test_data)
    print(net)
    # print(test_data)
    out = net(test_data)
    print(out.shape)
