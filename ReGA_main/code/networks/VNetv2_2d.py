import torch
from torch import nn
from torch.nn import init

# Weight initialization functions remain unchanged
# ...

class ResUnetConv2(nn.Module):
    def __init__(self, in_size, out_size, dropout_p, is_batchnorm, kernel_size=(3, 3), padding_size=(1, 1), init_stride=(1, 1)):
        super(ResUnetConv2, self).__init__()
        
        self.in_size = in_size
        self.out_size = out_size

        if in_size != out_size:
            self.conv_trans = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1),
                                            nn.InstanceNorm2d(out_size, affine=True),
                                            nn.LeakyReLU(inplace=True), )

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm2d(out_size, affine=True),
                                       nn.LeakyReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm2d(out_size, affine=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.LeakyReLU(inplace=True), )
            self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, 1, padding_size)

        self.dropout = nn.Dropout(dropout_p)
        self.activate = nn.LeakyReLU(inplace=True)

        # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)

        if self.in_size != self.out_size:
            inputs_trans = self.conv_trans(inputs)
        else:
            inputs_trans = inputs

        outputs = self.activate(inputs_trans + outputs)

        return outputs


class Encoder_dropout2D(nn.Module):
    def __init__(self, params):
        super(Encoder_dropout2D, self).__init__()
        self.params = params
        self.in_channels = self.params['in_chns']
        self.is_batchnorm = self.params['is_batchnorm']
        self.dropout = self.params['dropout']

        filters = self.params['filters']

        self.conv1 = ResUnetConv2(self.in_channels, filters[0], self.dropout[0], self.is_batchnorm)
        self.pool1 = nn.Sequential(nn.Conv2d(filters[0], filters[1], kernel_size=(2, 2), stride=(2, 2)),
                                   nn.InstanceNorm2d(filters[1], affine=True),
                                   nn.LeakyReLU(inplace=True))

        self.conv2 = ResUnetConv2(filters[1], filters[1], self.dropout[1], self.is_batchnorm)
        self.pool2 = nn.Sequential(nn.Conv2d(filters[1], filters[2], kernel_size=(2, 2), stride=(2, 2)),
                                   nn.InstanceNorm2d(filters[2], affine=True),
                                   nn.LeakyReLU(inplace=True))

        self.conv3 = ResUnetConv2(filters[2], filters[2], self.dropout[2], self.is_batchnorm)
        self.pool3 = nn.Sequential(nn.Conv2d(filters[2], filters[3], kernel_size=(2, 2), stride=(2, 2)),
                                   nn.InstanceNorm2d(filters[3], affine=True),
                                   nn.LeakyReLU(inplace=True))

        self.conv4 = ResUnetConv2(filters[3], filters[3], self.dropout[3], self.is_batchnorm)
        self.pool4 = nn.Sequential(nn.Conv2d(filters[3], filters[4], kernel_size=(2, 2), stride=(2, 2)),
                                   nn.InstanceNorm2d(filters[4], affine=True),
                                   nn.LeakyReLU(inplace=True))

        self.conv5 = ResUnetConv2(filters[4], filters[4], self.dropout[4], self.is_batchnorm)
        self.pool5 = nn.Sequential(nn.Conv2d(filters[4], filters[5], kernel_size=(2, 2), stride=(2, 2)),
                                   nn.InstanceNorm2d(filters[5], affine=True),
                                   nn.LeakyReLU(inplace=True))

        self.center = ResUnetConv2(filters[5], filters[5], 0, self.is_batchnorm)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.InstanceNorm2d):
        #         init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)
        pool5 = self.pool5(conv5)

        center = self.center(pool5)

        return [conv1, conv2, conv3, conv4, conv5, center]


class UnetConv2(nn.Module):
    def __init__(self, in_size, out_size, dropout_p, is_batchnorm, kernel_size=(3, 3), padding_size=(1, 1), init_stride=(1, 1)):
        super(UnetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm2d(out_size, affine=True),
                                       nn.LeakyReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm2d(out_size, affine=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.LeakyReLU(inplace=True), )
            self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, 1, padding_size)

        self.dropout = nn.Dropout(dropout_p)

        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp2_CT_HM(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, dropout_p=0.5, up_factor=(2, 2), is_batchnorm=True):
        super(UnetUp2_CT_HM, self).__init__()

        self.up = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=up_factor, stride=up_factor),
                                nn.InstanceNorm2d(out_size, affine=True),
                                nn.LeakyReLU(inplace=True))

        self.conv = UnetConv2(out_size * 2, out_size, dropout_p, is_batchnorm, kernel_size=kernel_size,
                              padding_size=[(i - 1) // 2 for i in kernel_size])
        self.activate = nn.LeakyReLU(inplace=True)

        # for m in self.children():
        #     if m.__class__.__name__.find('UnetConv2') != -1:
        #         continue
        #     init_weights(m, init_type='kaiming')

    def forward(self, input1, input2):
        output2 = self.up(input2)
        x = torch.cat([input1, output2], 1)
        outputs = self.conv(x)
        outputs = self.activate(outputs + output2)
        return outputs


class Decoder_dropout2D(nn.Module):
    def __init__(self, params):
        super(Decoder_dropout2D, self).__init__()
        self.params = params
        self.is_batchnorm = self.params['is_batchnorm']
        self.dropout_p = self.params['dropout']

        filters = self.params['filters']

        self.up_concat5 = UnetUp2_CT_HM(filters[5], filters[4], kernel_size=(3, 3), up_factor=(2, 2),
                                        dropout_p=self.dropout_p[0], is_batchnorm=self.is_batchnorm)
        self.up_concat4 = UnetUp2_CT_HM(filters[4], filters[3], kernel_size=(3, 3), up_factor=(2, 2),
                                        dropout_p=self.dropout_p[1], is_batchnorm=self.is_batchnorm)
        self.up_concat3 = UnetUp2_CT_HM(filters[3], filters[2], kernel_size=(3, 3), up_factor=(2, 2),
                                        dropout_p=self.dropout_p[2], is_batchnorm=self.is_batchnorm)
        self.up_concat2 = UnetUp2_CT_HM(filters[2], filters[1], kernel_size=(3, 3), up_factor=(2, 2),
                                        dropout_p=self.dropout_p[3], is_batchnorm=self.is_batchnorm)
        self.up_concat1 = UnetUp2_CT_HM(filters[1], filters[0], kernel_size=(3, 3), up_factor=(2, 2),
                                        dropout_p=self.dropout_p[4], is_batchnorm=self.is_batchnorm)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.InstanceNorm2d):
        #         init_weights(m, init_type='kaiming')

    def forward(self, feature):
        conv1 = feature[0]
        conv2 = feature[1]
        conv3 = feature[2]
        conv4 = feature[3]
        conv5 = feature[4]
        center = feature[5]

        up5 = self.up_concat5(conv5, center)
        up4 = self.up_concat4(conv4, up5)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        return up1


class VNet_CCT_dropout_2D(nn.Module):
    def __init__(self, in_channels=1, n_classes=2,Tanh_gene = True,is_batchnorm=True):
        super(VNet_CCT_dropout_2D, self).__init__()
        print("Using VNetv2 2D")

        params_encoder = {
            'in_chns': in_channels,
            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
            'is_batchnorm': is_batchnorm,
            'filters': [16, 32, 64, 128, 256, 512]
        }

        params_decoder_main = {
            'dropout': [0, 0, 0, 0, 0],
            'is_batchnorm': is_batchnorm,
            'filters': [16, 32, 64, 128, 256, 512]
        }

        self.encoder = Encoder_dropout2D(params_encoder)
        self.main_decoder_1 = Decoder_dropout2D(params_decoder_main)
        self.n_class = n_classes
        self.main_final_1 = nn.Conv2d(16, self.n_class, 1)

        self.Tanh_gene_bool = Tanh_gene
        self.Tanh_gene = nn.Tanh()
        self.Sigmoid_gene = nn.Sigmoid()

    def forward(self, x):
        feature_0 = self.encoder(x)
        main_outfeature_1 = self.main_decoder_1(feature_0)
        main_seg_1 = self.main_final_1(main_outfeature_1)
        if self.Tanh_gene_bool:
            main_seg_1 = self.Tanh_gene(main_seg_1)
        else:
            main_seg_1 = self.Sigmoid_gene(main_seg_1)
        return main_seg_1
