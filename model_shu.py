import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

def pixel_shuffle(input, scale_factor):
    batch_size, in_channels, in_height, in_width = input.size()

    out_channels = int(in_channels // (scale_factor * scale_factor))
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(
            batch_size, channels, upscale_factor, upscale_factor,
            in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)


        input_view = input.contiguous().view(
            batch_size, in_channels, out_height, block_size,
            out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm2d(out_channels)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(out_channels)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True, activation='lrelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(output_channels)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_channels)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()


    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out


class ResidualDenseBlock_5C(nn.Module):

    def __init__(self, input_channel, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = ConvBlock(input_channel, num_filter, kernel_size, stride, padding, bias, activation, norm)
        self.conv2= ConvBlock(input_channel+1*num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm)
        self.conv3= ConvBlock(input_channel+2*num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm)
        self.conv4= ConvBlock(input_channel+3*num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm)
        self.conv5= ConvBlock(input_channel+4*num_filter, input_channel, kernel_size, stride, padding, bias, None, norm)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x

class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, input_channel, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(input_channel, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None)
        self.RDB2 = ResidualDenseBlock_5C(input_channel, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None)
        self.RDB3 = ResidualDenseBlock_5C(input_channel, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x

class U_net1(nn.Module):
    def __init__(self, opts):
        super(U_net1, self).__init__()
        self.nf = opts.nf

        self.conv1_1 = ConvBlock(6, self.nf * 1, kernel_size = 3, stride = 1, padding = 1)
        self.conv1_2 = ResnetBlock(self.nf * 1)
        self.conv1_3 = ResnetBlock(self.nf * 1)
        self.conv1_4 = ResnetBlock(self.nf * 1)
        self.conv2_1 = ConvBlock(self.nf * 1, self.nf * 2, kernel_size = 4, stride = 2, padding = 1, bias=True)
        self.conv2_2 = ResnetBlock(self.nf * 2)
        self.conv2_3 = ResnetBlock(self.nf * 2)
        self.conv2_4 = ResnetBlock(self.nf * 2)
        self.conv3_1 = ConvBlock(self.nf * 2, self.nf * 4, kernel_size = 4, stride = 2, padding = 1, bias=True)
        self.conv3_2 = ResnetBlock(self.nf * 4)
        self.conv3_3 = ResnetBlock(self.nf * 4)
        self.conv3_4 = ResnetBlock(self.nf * 4)
        self.conv4 = ConvBlock(self.nf * 4, self.nf * 4, kernel_size = 3, stride = 1, padding = 1, bias=True)
        self.deconv3_4 = ResnetBlock(self.nf * 4)
        self.deconv3_3 = ResnetBlock(self.nf * 4)
        self.deconv3_2 = ResnetBlock(self.nf * 4)
        self.deconv3_1 = DeconvBlock(self.nf * 4, self.nf * 2, kernel_size = 4, stride = 2, padding = 1, output_padding=(0,0), bias=True)
        self.deconv2_4 = ResnetBlock(self.nf * 4)
        self.deconv2_3 = ResnetBlock(self.nf * 4)
        self.deconv2_2 = ResnetBlock(self.nf * 4)
        self.deconv2_1 = DeconvBlock(self.nf * 4, self.nf * 1, kernel_size = 4, stride = 2, padding = 1, bias=True)
        self.deconv1_4 = ResnetBlock(self.nf * 2)
        self.deconv1_3 = ResnetBlock(self.nf * 2)
        self.deconv1_2 = ResnetBlock(self.nf * 2)
        self.deconv1_1 = ConvBlock(self.nf * 2, 2, kernel_size = 3, stride = 1, padding = 1, bias=True)


    def forward(self, x, y):
        input = torch.cat((x, y), 1)
        out_conv1_1 = self.conv1_1(input)
        out_conv1_2 = self.conv1_2(out_conv1_1)
        out_conv1_3 = self.conv1_3(out_conv1_2)
        out_conv1_4 = self.conv1_4(out_conv1_3)
        out_conv2_1 = self.conv2_1(out_conv1_4)
        out_conv2_2 = self.conv2_2(out_conv2_1)
        out_conv2_3 = self.conv2_3(out_conv2_2)
        out_conv2_4 = self.conv2_4(out_conv2_3)
        out_conv3_1 = self.conv3_1(out_conv2_4)

        out_conv3_2 = self.conv3_2(out_conv3_1)
        out_conv3_3 = self.conv3_3(out_conv3_2)
        out_conv3_4 = self.conv3_4(out_conv3_3)
        out_conv4 = self.conv4(out_conv3_4)
        out_deconv3_4 = self.deconv3_4(out_conv4)
        out_deconv3_3 = self.deconv3_3(out_deconv3_4)
        out_deconv3_2 = self.deconv3_2(out_deconv3_3)
        out_deconv3_1 = self.deconv3_1(out_deconv3_2)
        out_deconv3_1 = torch.cat((out_deconv3_1, out_conv2_4), 1)
        out_deconv2_4 = self.deconv2_4(out_deconv3_1)
        out_deconv2_3 = self.deconv2_3(out_deconv2_4)
        out_deconv2_2 = self.deconv2_2(out_deconv2_3)
        out_deconv2_1 = self.deconv2_1(out_deconv2_2)
        out_deconv2_1 = torch.cat((out_deconv2_1, out_conv1_4), 1)
        out_deconv1_4 = self.deconv1_4(out_deconv2_1)
        out_deconv1_3 = self.deconv1_3(out_deconv1_4)
        out_deconv1_2 = self.deconv1_2(out_deconv1_3)
        out_deconv1_1 = self.deconv1_1(out_deconv1_2)

        return out_deconv1_1


class U_net2(nn.Module):
    def __init__(self, opts):
        super(U_net2, self).__init__()
        self.nf = opts.nf
        

        # U-net part
        self.conv1_1 = ConvBlock(3+48, self.nf * 4, kernel_size = 3, stride = 1, padding = 1, bias=True)
        self.conv1_2 = ResnetBlock(self.nf * 4)
        self.conv1_3 = ResnetBlock(self.nf * 4)
        self.conv1_4 = ResnetBlock(self.nf * 4)

        self.deconv3_7 = ResnetBlock(self.nf * 4)
        self.deconv3_6 = ResnetBlock(self.nf * 4)
        self.deconv3_5 = ResnetBlock(self.nf * 4)
        self.deconv3_4 = ResnetBlock(self.nf * 4)
        self.deconv3_3 = ResnetBlock(self.nf * 4)
        self.deconv3_2 = ResnetBlock(self.nf * 4)
        self.deconv3_1 = DeconvBlock(self.nf * 4, self.nf * 2, kernel_size = 4, stride = 2, padding = 1, bias=True)
        self.deconv2_4 = ResnetBlock(self.nf * 2)
        self.deconv2_3 = ResnetBlock(self.nf * 2)
        self.deconv2_2 = ResnetBlock(self.nf * 2)
        self.deconv2_1 = DeconvBlock(self.nf * 2, self.nf * 1, kernel_size = 4, stride = 2, padding = 1, bias=True)
        self.deconv1_4 = ResnetBlock(self.nf * 1)
        self.deconv1_3 = ResnetBlock(self.nf * 1)
        self.deconv1_2 = ResnetBlock(self.nf * 1)
        self.deconv1_1 = ConvBlock(self.nf * 1, 3, kernel_size = 3, stride = 1, padding = 1, bias=True)

        self.deconv3_7_r = ResnetBlock(self.nf * 4)
        self.deconv3_6_r = ResnetBlock(self.nf * 4)
        self.deconv3_5_r = ResnetBlock(self.nf * 4)
        self.deconv3_4_r = ResnetBlock(self.nf * 4)
        self.deconv3_3_r = ResnetBlock(self.nf * 4)
        self.deconv3_2_r = ResnetBlock(self.nf * 4)
        self.deconv3_1_r = DeconvBlock(self.nf * 4, self.nf * 2, kernel_size = 4, stride = 2, padding = 1, bias=True)
        self.deconv2_4_r = ResnetBlock(self.nf * 2)
        self.deconv2_3_r = ResnetBlock(self.nf * 2)
        self.deconv2_2_r = ResnetBlock(self.nf * 2)
        self.deconv2_1_r = DeconvBlock(self.nf * 2, self.nf * 1, kernel_size = 4, stride = 2, padding = 1, bias=True)
        self.deconv1_4_r = ResnetBlock(self.nf * 1)
        self.deconv1_3_r = ResnetBlock(self.nf * 1)
        self.deconv1_2_r = ResnetBlock(self.nf * 1)
        self.deconv1_1_r = ConvBlock(self.nf * 1, 3, kernel_size = 3, stride = 1, padding = 1, bias=True)       
        

    def forward(self, x, y):
        input = pixel_shuffle(x, 1/4)
        input = torch.cat((input, y), 1)

        out_conv1_1 = self.conv1_1(input)
        out_conv1_2 = self.conv1_2(out_conv1_1)
        out_conv1_3 = self.conv1_3(out_conv1_2)
        out_conv1_4 = self.conv1_4(out_conv1_3)

        out_deconv3_7 = self.deconv3_7(out_conv1_4)
        out_deconv3_6 = self.deconv3_6(out_deconv3_7)
        out_deconv3_5 = self.deconv3_5(out_deconv3_6)
        out_deconv3_4 = self.deconv3_4(out_deconv3_5)
        out_deconv3_3 = self.deconv3_3(out_deconv3_4)
        out_deconv3_2 = self.deconv3_2(out_deconv3_3)
        out_deconv3_2 = out_deconv3_2 + out_conv1_4
        out_deconv3_1 = self.deconv3_1(out_deconv3_2)

        out_deconv2_4 = self.deconv2_4(out_deconv3_1)
        out_deconv2_3 = self.deconv2_3(out_deconv2_4)
        out_deconv2_2 = self.deconv2_2(out_deconv2_3)
        out_deconv2_1 = self.deconv2_1(out_deconv2_2)

        out_deconv1_4 = self.deconv1_4(out_deconv2_1)
        out_deconv1_3 = self.deconv1_3(out_deconv1_4)
        out_deconv1_2 = self.deconv1_2(out_deconv1_3)
        out_deconv1_1 = self.deconv1_1(out_deconv1_2)

        out_deconv3_7_r = self.deconv3_7_r(out_conv1_4)
        out_deconv3_6_r = self.deconv3_6_r(out_deconv3_7_r)
        out_deconv3_5_r = self.deconv3_5_r(out_deconv3_6_r)
        out_deconv3_4_r = self.deconv3_4_r(out_deconv3_5_r)
        out_deconv3_3_r = self.deconv3_3_r(out_deconv3_4_r)
        out_deconv3_2_r = self.deconv3_2_r(out_deconv3_3_r)
        out_deconv3_2_r = out_deconv3_2_r + out_conv1_4
        out_deconv3_1_r = self.deconv3_1_r(out_deconv3_2_r)

        out_deconv2_4_r = self.deconv2_4_r(out_deconv3_1_r)
        out_deconv2_3_r = self.deconv2_3_r(out_deconv2_4_r)
        out_deconv2_2_r = self.deconv2_2_r(out_deconv2_3_r)
        out_deconv2_1_r = self.deconv2_1_r(out_deconv2_2_r)

        out_deconv1_4_r = self.deconv1_4_r(out_deconv2_1_r)
        out_deconv1_3_r = self.deconv1_3_r(out_deconv1_4_r)
        out_deconv1_2_r = self.deconv1_2_r(out_deconv1_3_r)
        out_deconv1_1_r = self.deconv1_1_r(out_deconv1_2_r) 

        return out_deconv1_1, out_deconv1_1_r

class downsample(nn.Module):
    def __init__(self, opts):
        super(downsample, self).__init__()

        self.nf = opts.nf


       
        self.conv1_1 = ConvBlock(3, self.nf * 1, kernel_size = 3, stride = 1, padding = 1, bias=True)
        self.conv1_2 = ResnetBlock(self.nf * 1)
        self.conv1_3 = ResnetBlock(self.nf * 1)
        self.conv1_4 = ResnetBlock(self.nf * 1)
        self.conv1_5 = ResnetBlock(self.nf * 1)
        self.conv1_6 = ResnetBlock(self.nf * 1)
        self.conv1_7 = ResnetBlock(self.nf * 1)
        self.conv2_1 = ConvBlock(self.nf * 1, self.nf * 2, kernel_size = 4, stride = 2, padding = 1, bias=True)
        self.conv2_2 = ResnetBlock(self.nf * 2)
        self.conv2_3 = ResnetBlock(self.nf * 2)
        self.conv2_4 = ResnetBlock(self.nf * 2)
        self.conv3_1 = ConvBlock(self.nf * 2, self.nf * 4, kernel_size = 4, stride = 2, padding = 1, bias=True)
        self.conv3_2 = ResnetBlock(self.nf * 4)
        self.conv3_3 = ResnetBlock(self.nf * 4)
        self.conv3_4 = ResnetBlock(self.nf * 4)
        self.conv4 = ConvBlock(self.nf * 4, 3, kernel_size = 3, stride = 1, padding = 1, bias=True)



    def forward(self, x):
        input = x
        out_conv1_1 = self.conv1_1(input)
        out_conv1_2 = self.conv1_2(out_conv1_1)
        out_conv1_3 = self.conv1_3(out_conv1_2)
        out_conv1_4 = self.conv1_4(out_conv1_3)
        out_conv1_5 = self.conv1_5(out_conv1_4)
        out_conv1_6 = self.conv1_6(out_conv1_5)
        out_conv1_7 = self.conv1_7(out_conv1_6)
        out_conv1_7 = out_conv1_7 + out_conv1_1
        out_conv2_1 = self.conv2_1(out_conv1_7)
        out_conv2_2 = self.conv2_2(out_conv2_1)
        out_conv2_3 = self.conv2_3(out_conv2_2)
        out_conv2_4 = self.conv2_4(out_conv2_3)
        out_conv3_1 = self.conv3_1(out_conv2_4)

        out_conv3_2 = self.conv3_2(out_conv3_1)
        out_conv3_3 = self.conv3_3(out_conv3_2)
        out_conv3_4 = self.conv3_4(out_conv3_3)
        out_conv4 = self.conv4(out_conv3_4)


        return out_conv4

