import logging
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from MinkowskiEngine import utils as ME_utils
from datetime import datetime

current_date = datetime.now()
formatted_date = str(current_date.strftime("%Y%m%d"))

task_name = 'austin_network_{}'.format(formatted_date)
log_file = task_name + '.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class MyInception_1(nn.Module):
    def __init__(self,
                 channels,
                 stride=1,
                 dilation=1,
                 bn_momentum=0.1,
                 dimension=3):
        super(MyInception_1, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=1, stride=stride, dilation=dilation, bias=True, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            channels // 4, channels // 4, kernel_size=3, stride=stride, dilation=dilation, bias=True,
            dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.conv3 = ME.MinkowskiConvolution(
            channels // 4, channels // 2, kernel_size=1, stride=stride, dilation=dilation, bias=True,
            dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(channels // 2, momentum=bn_momentum)

        self.conv4 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=3, stride=stride, dilation=dilation, bias=True, dimension=dimension)
        self.norm4 = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.conv5 = ME.MinkowskiConvolution(
            channels // 4, channels // 2, kernel_size=3, stride=stride, dilation=dilation, bias=True,
            dimension=dimension)
        self.norm5 = ME.MinkowskiBatchNorm(channels // 2, momentum=bn_momentum)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # 1
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu(out)

        # 2
        out1 = self.conv4(x)
        out1 = self.norm4(out1)
        out1 = self.relu(out1)

        out1 = self.conv5(out1)
        out1 = self.norm5(out1)
        out1 = self.relu(out1)

        # 3
        out2 = ME.cat(out, out1)
        out2 += x

        return out2


class Pyramid_1(nn.Module):
    def __init__(self,
                 channels,
                 bn_momentum=0.1,
                 dimension=3):
        super(Pyramid_1, self).__init__()
        assert dimension > 0

        self.aspp1 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=1, stride=1, dilation=1, bias=True, dimension=dimension)
        self.aspp2 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=3, stride=1, dilation=6, bias=True, dimension=dimension)
        self.aspp3 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=3, stride=1, dilation=12, bias=True, dimension=dimension)
        self.aspp4 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=3, stride=1, dilation=18, bias=True, dimension=dimension)
        self.aspp5 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=1, stride=1, dilation=1, bias=True, dimension=dimension)

        self.aspp1_bn = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.aspp2_bn = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.aspp3_bn = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.aspp4_bn = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.aspp5_bn = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)

        self.conv2 = ME.MinkowskiConvolution(
            channels // 4 * 5, channels, kernel_size=1, stride=1, dilation=1, bias=True, dimension=dimension)
        self.bn2 = ME.MinkowskiBatchNorm(channels, momentum=bn_momentum)

        self.pooling = ME.MinkowskiGlobalPooling()
        self.broadcast = ME.MinkowskiBroadcast()
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)

        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)

        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)

        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)

        x5 = self.pooling(x)
        x5 = self.broadcast(x, x5)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)

        x6 = ME.cat(x1, x2, x3, x4, x5)
        x6 = self.conv2(x6)
        x6 = self.bn2(x6)
        x6 = self.relu(x6)

        x7 = x6 + x

        return x7


class MyNet(ME.MinkowskiNetwork):
    CHANNELS = [None, 32, 32, 64, 128, 256]
    TR_CHANNELS = [None, 32, 32, 64, 128, 256]
    BLOCK_1 = MyInception_1
    BLOCK_2 = Pyramid_1

    def __init__(self,
                 in_channels=3,  # 修改输入通道数为3
                 out_channels=3,  # 修改输出通道数为3
                 bn_momentum=0.1,
                 D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        CHANNELS = self.CHANNELS
        TR_CHANNELS = self.TR_CHANNELS
        BLOCK_1 = self.BLOCK_1
        BLOCK_2 = self.BLOCK_2

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[1],
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm1 = ME.MinkowskiBatchNorm(CHANNELS[1], momentum=bn_momentum)
        self.block1 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2 = ME.MinkowskiBatchNorm(CHANNELS[2], momentum=bn_momentum)
        self.block2 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3 = ME.MinkowskiBatchNorm(CHANNELS[3], momentum=bn_momentum)
        self.block3 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv4 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[3],
            out_channels=CHANNELS[4],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4 = ME.MinkowskiBatchNorm(CHANNELS[4], momentum=bn_momentum)
        self.block4 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv5 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[4],
            out_channels=CHANNELS[5],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm5 = ME.MinkowskiBatchNorm(CHANNELS[5], momentum=bn_momentum)
        self.block5 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[5], bn_momentum=bn_momentum, D=D)

        self.conv5_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[5],
            out_channels=TR_CHANNELS[5],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm5_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[5], momentum=bn_momentum)
        self.block5_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[5], bn_momentum=bn_momentum, D=D)

        self.conv4_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[4] + TR_CHANNELS[5],
            out_channels=TR_CHANNELS[4],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[4], momentum=bn_momentum)
        self.block4_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv3_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[3] + TR_CHANNELS[4],
            out_channels=TR_CHANNELS[3],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[3], momentum=bn_momentum)
        self.block3_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[2] + TR_CHANNELS[3],
            out_channels=TR_CHANNELS[2],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[2], momentum=bn_momentum)
        self.block2_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv1_tr = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        self.final = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=D)

    def make_layer(self, block_1, block_2, channels, bn_momentum, D):
        layers = []
        # MyInception
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))
        # Pyramid_1
        layers.append(block_2(channels=channels, bn_momentum=bn_momentum))
        # MyInception
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 预测值特别大  e10
        # print("Input mean:", x.F.mean().item())
        # print("Input std:", x.F.std().item())
        # print("Input max:", x.F.max().item())
        # print("Input min:", x.F.min().item())
        # logger.info("Input mean: %s", x.F.mean().item())
        # logger.info("Input std: %s", x.F.std().item())
        # logger.info("Input max: %s", x.F.max().item())
        # logger.info("Input min: %s", x.F.min().item())

        # 使用初始坐标映射键进行所有输出
        # print("Input shape:", x.shape)
        out_s1 = self.conv1(x)
        # print("After conv1 shape:", out_s1.shape)
        out_s1 = self.norm1(out_s1)
        out_s1 = self.block1(out_s1)
        out = MEF.relu(out_s1)

        # print("After block1 shape:", out.shape)
        # print("After block1 coordinate_map_key:", out.coordinate_map_key)

        out_s2 = self.conv2(out)
        # print("After conv2 shape:", out_s2.shape)
        out_s2 = self.norm2(out_s2)
        out_s2 = self.block2(out_s2)
        out = MEF.relu(out_s2)

        # print("After block2 shape:", out.shape)
        # print("After block2 coordinate_map_key:", out.coordinate_map_key)

        out_s4 = self.conv3(out)
        # print("After conv3 shape:", out_s4.shape)
        out_s4 = self.norm3(out_s4)
        out_s4 = self.block3(out_s4)
        out = MEF.relu(out_s4)

        # print("After block3 shape:", out.shape)
        # print("After block3 coordinate_map_key:", out.coordinate_map_key)

        out_s8 = self.conv4(out)
        # print("After conv4 shape:", out_s8.shape)
        out_s8 = self.norm4(out_s8)
        out_s8 = self.block4(out_s8)
        out = MEF.relu(out_s8)

        # print("After block4 shape:", out.shape)
        # print("After block4 coordinate_map_key:", out.coordinate_map_key)

        out_s16 = self.conv5(out)
        # print("After conv5 shape:", out_s16.shape)
        out_s16 = self.norm5(out_s16)
        out_s16 = self.block5(out_s16)
        out = MEF.relu(out_s16)

        # print("After block5 shape:", out.shape)
        # print("After block5 coordinate_map_key:", out.coordinate_map_key)

        out = self.conv5_tr(out)
        out = self.norm5_tr(out)
        out = self.block5_tr(out)
        out_s8_tr = MEF.relu(out)

        # print("After block5_tr shape:", out.shape)
        # print("After block5_tr coordinate_map_key:", out.coordinate_map_key)

        # out_s8_tr.coordinate_map_key = initial_coordinate_map_key
        # out_s8.coordinate_map_key = initial_coordinate_map_key

        # print('DEBUG: ',out_s8_tr.coordinate_map_key,out_s8.coordinate_map_key,out_s8_tr.shape,out_s8.shape)
        out = ME.cat(out_s8_tr, out_s8)

        # print("After concatenating out_s8_tr and out_s8 shape:", out.shape)
        # print("After concatenating out_s8_tr and out_s8 coordinate_map_key:", out.coordinate_map_key)

        out = self.conv4_tr(out)
        out = self.norm4_tr(out)
        out = self.block4_tr(out)
        out_s4_tr = MEF.relu(out)

        # print("After block4_tr shape:", out.shape)
        # print("After block4_tr coordinate_map_key:", out.coordinate_map_key)

        # print('DEBUG: ',out_s4_tr.coordinate_map_key,out_s4.coordinate_map_key)
        out = ME.cat(out_s4_tr, out_s4)

        # print("After concatenating out_s4_tr and out_s4 shape:", out.shape)
        # print("After concatenating out_s4_tr and out_s4 coordinate_map_key:", out.coordinate_map_key)

        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out = self.block3_tr(out)
        out_s2_tr = MEF.relu(out)

        # print("After block3_tr shape:", out.shape)
        # print("After block3_tr coordinate_map_key:", out.coordinate_map_key)

        out = ME.cat(out_s2_tr, out_s2)

        # print("After concatenating out_s2_tr and out_s2 shape:", out.shape)
        # print("After concatenating out_s2_tr and out_s2 coordinate_map_key:", out.coordinate_map_key)

        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out = self.block2_tr(out)
        out_s1_tr = MEF.relu(out)

        # print("After block2_tr shape:", out.shape)
        # print("After block2_tr coordinate_map_key:", out.coordinate_map_key)

        out = out_s1_tr + out_s1

        # print("After adding out_s1_tr and out_s1 shape:", out.shape)
        # print("After adding out_s1_tr and out_s1 coordinate_map_key:", out.coordinate_map_key)

        out = self.conv1_tr(out)
        out = MEF.relu(out)


        # print("After conv1_tr shape:", out.shape)
        # print("After conv1_tr coordinate_map_key:", out.coordinate_map_key)

        out_cls = self.final(out)
        # print("After final shape:", out_cls.shape)
        # print("After final coordinate_map_key:", out_cls.coordinate_map_key)
        # print("After final mean:", out_cls.F.mean().item())
        # print("After final std:", out_cls.F.std().item())
        # print("After final max:", out_cls.F.max().item())
        # print("After final min:", out_cls.F.min().item())
        # print('\n\n')
        # logger.info("After final mean: %s", out_cls.F.mean().item())
        # logger.info("After final std: %s", out_cls.F.std().item())
        # logger.info("After final max: %s", out_cls.F.max().item())
        # logger.info("After final min: %s", out_cls.F.min().item())
        # logger.info("\n")

        return out_cls
