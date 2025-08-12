import torch
import torch.nn as nn
from .net_utils import *
from .net_utils import _DenseASPPBlock
import torch.nn.functional as F

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
 

class encoder(nn.Module):
    def __init__(self, params):
        super(encoder, self).__init__()
        self.params = params
        import torchvision.models as models
        if params.encoder == 'densenet121_bts':
            self.base_model = models.densenet121(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [64, 64, 128, 256, 1024]
        elif params.encoder == 'densenet161_bts':
            self.base_model = models.densenet161(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [96, 96, 192, 384, 2208]
        elif params.encoder == 'resnet50_bts':
            self.base_model = models.resnet50(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnet34_bts':
            self.base_model = models.resnet34(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        elif params.encoder == 'resnet18_bts':
            self.base_model = models.resnet18(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        elif params.encoder == 'resnet101_bts':
            self.base_model = models.resnet101(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext50_bts':
            self.base_model = models.resnext50_32x4d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext101_bts':
            self.base_model = models.resnext101_32x8d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'mobilenetv2_bts':
            self.base_model = models.mobilenet_v2(pretrained=True).features
            self.feat_inds = [2, 4, 7, 11, 19]
            self.feat_out_channels = [16, 24, 32, 64, 1280]
            self.feat_names = []
        else:
            print('Not supported encoder: {}'.format(params.encoder))

    def forward(self, x):
        feature = x
        skip_feat = []
        i = 1
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if self.params.encoder == 'mobilenetv2_bts':
                if i == 2 or i == 4 or i == 7 or i == 11 or i == 19:
                    skip_feat.append(feature)
            else:
                if any(x in k for x in self.feat_names):
                    skip_feat.append(feature)
            i = i + 1
        return skip_feat


class Unet_decoder(nn.Module):
    def __init__(self, 
                 params,
                 n_filters,
                 n_skips_image,
                 n_skips_radar,
                 activation_func=torch.nn.ReLU(inplace=False),
                 use_batch_norm=False):
        super(Unet_decoder, self).__init__()
        self.params = params
        
        filter_idx = 0
        network_depth = len(n_filters)
        # if len(n_skips_image) < network_depth:
        # means that image features start from 1/4 not 1/2

        # Resolution 1/32 -> 1/16

        if params.fuse == 'concat':
            in_channels=n_skips_image[-1]+n_skips_radar[-1]
            skip_channels=n_skips_image[-2]+n_skips_radar[-2]

        elif params.fuse == 'gated_fuse':
            self.weight4    = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-1], n_skips_image[-1], 1, 1, bias=False),
                                              nn.Sigmoid())
            self.project4   = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-1], n_skips_image[-1], 1, 1, bias=False),
                                              nn.ReLU())
            in_channels=n_skips_image[-1]
            skip_channels=n_skips_image[-2]  
 
        elif params.fuse == 'wafb':
            self.weight4    = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-1], n_skips_image[-1], 1, 1, bias=False),
                                              nn.Sigmoid())
            self.project4   = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-1], n_skips_image[-1], 1, 1, bias=False),
                                              nn.ReLU())
            self.gate4 = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-1]+self.params.text_hidden_dim, n_skips_image[-1], 1, 1, bias=False),
                                              nn.Sigmoid())
            in_channels=n_skips_image[-1]
            skip_channels=n_skips_image[-2]  
        else:
            print('Not supported.')


        self.deconv4 = DecoderBlock(
            in_channels=in_channels,
            skip_channels=skip_channels,
            out_channels=n_filters[filter_idx],
            activation_func=activation_func
        )
        filter_idx += 1

        # add DenseASPP
        self.aspp = _DenseASPPBlock(
            in_channels=n_filters[0],
            inter_channels1=64,
            inter_channels2=64,
        )
        self.daspp_conv = Conv2d(
            in_channels=n_filters[0]+5*64,
            out_channels=n_filters[0],
            kernel_size=3,
            stride=1,
            activation_func=activation_func
        )
        
        # Resolution 1/16 -> 1/8
        if params.fuse == 'concat':
            in_channels=n_filters[0]
            skip_channels=n_skips_image[-3]+n_skips_radar[-3]

        elif params.fuse == 'gated_fuse':
            self.weight3    = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-2], n_filters[0], 1, 1, bias=False),
                                              nn.Sigmoid())
            self.project3   = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-2], n_filters[0], 1, 1, bias=False),
                                              nn.ReLU())
            in_channels=n_filters[0]
            skip_channels=n_skips_image[-3]  

        elif params.fuse == 'wafb':
            self.weight3    = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-2], n_filters[0], 1, 1, bias=False),
                                              nn.Sigmoid())
            self.project3   = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-2], n_filters[0], 1, 1, bias=False),
                                              nn.ReLU())
            self.gate3 = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-2]+self.params.text_hidden_dim, n_filters[0], 1, 1, bias=False),
                                              nn.Sigmoid())
            in_channels=n_filters[0]
            skip_channels=n_skips_image[-3]  
        else:
            print('Not supported.')

        self.deconv3 = DecoderBlock(
            in_channels=in_channels,
            skip_channels=skip_channels,
            out_channels=n_filters[filter_idx],
            activation_func=activation_func
        )
        filter_idx += 1

        # Resolution 1/8 -> 1/4
        if params.fuse == 'concat':
            in_channels=n_filters[filter_idx-1]
            skip_channels=n_skips_image[-4]+n_skips_radar[-4]

        elif params.fuse == 'gated_fuse':
            self.weight2    = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-3], n_filters[filter_idx-1], 1, 1, bias=False),
                                              nn.Sigmoid())
            self.project2   = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-3], n_filters[filter_idx-1], 1, 1, bias=False),
                                              nn.ReLU())
            in_channels=n_filters[filter_idx-1]
            skip_channels=n_skips_image[-4]  

        elif params.fuse == 'wafb':
            self.weight2    = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-3], n_filters[filter_idx-1], 1, 1, bias=False),
                                              nn.Sigmoid())
            self.project2   = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-3], n_filters[filter_idx-1], 1, 1, bias=False),
                                              nn.ReLU())
            self.gate2 = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-3]+self.params.text_hidden_dim, n_filters[filter_idx-1], 1, 1, bias=False),
                                              nn.Sigmoid())
            in_channels=n_filters[filter_idx-1]
            skip_channels=n_skips_image[-4]  
        else:
            print('Not supported.')

        self.deconv2 = DecoderBlock(
            in_channels=in_channels,
            skip_channels=skip_channels,
            out_channels=n_filters[filter_idx],
            activation_func=activation_func
        )
        filter_idx += 1

        # Resolution 1/4 -> 1/2

        if len(n_skips_image) < 5:
            if params.fuse == 'concat':
                in_channels=n_filters[filter_idx-1]
                skip_channels=n_skips_radar[-5]
  
            elif params.fuse == 'gated_fuse':
                self.weight1    = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-4], n_filters[filter_idx-1], 1, 1, bias=False),
                                                nn.Sigmoid())
                self.project1   = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-4], n_filters[filter_idx-1], 1, 1, bias=False),
                                                nn.ReLU())
                in_channels=n_filters[filter_idx-1]
                skip_channels=0

            elif params.fuse == 'wafb':
                self.weight1    = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-4], n_filters[filter_idx-1], 1, 1, bias=False),
                                                nn.Sigmoid())
                self.project1   = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-4], n_filters[filter_idx-1], 1, 1, bias=False),
                                                nn.ReLU())
                self.gate1 = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-4]+self.params.text_hidden_dim, n_filters[filter_idx-1], 1, 1, bias=False),
                                              nn.Sigmoid())
                in_channels=n_filters[filter_idx-1]
                skip_channels=0
            else:
                print('Not supported.')

        else:
            if params.fuse == 'concat':
                in_channels=n_filters[filter_idx-1]
                skip_channels=n_skips_image[-5]+n_skips_radar[-5]

            elif params.fuse == 'gated_fuse':
                self.weight1    = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-4], n_filters[filter_idx-1], 1, 1, bias=False),
                                                      nn.Sigmoid())
                self.project1   = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-4], n_filters[filter_idx-1], 1, 1, bias=False),
                                                      nn.ReLU())
                in_channels=n_filters[filter_idx-1]
                skip_channels=n_skips_image[-5]  

            elif params.fuse == 'wafb':
                self.weight1    = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-4], n_filters[filter_idx-1], 1, 1, bias=False),
                                                nn.Sigmoid())
                self.project1   = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-4], n_filters[filter_idx-1], 1, 1, bias=False),
                                                      nn.ReLU())
                self.gate1 = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-4]+self.params.text_hidden_dim, n_filters[filter_idx-1], 1, 1, bias=False),
                                                 nn.Sigmoid())
                in_channels=n_filters[filter_idx-1]
                skip_channels=n_skips_image[-5]  
            else:
                print('Not supported.')
        

        self.deconv1 = DecoderBlock(
            in_channels=in_channels,
            skip_channels=skip_channels,
            out_channels=n_filters[filter_idx],
            activation_func=activation_func
            )
        filter_idx += 1

        # Resolution 1/2 -> 1
        if params.fuse == 'concat':
            in_channels=n_filters[filter_idx-1]
            skip_channels=0

        elif params.fuse == 'gated_fuse':
            self.weight0    = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-5], n_filters[filter_idx-1], 1, 1, bias=False),
                                              nn.Sigmoid())
            self.project0   = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-5], n_filters[filter_idx-1], 1, 1, bias=False),
                                              nn.ReLU())
            in_channels=n_filters[filter_idx-1]
            skip_channels=0

        elif params.fuse == 'wafb':
            self.weight0    = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-5], n_filters[filter_idx-1], 1, 1, bias=False),
                                              nn.Sigmoid())
            self.project0   = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-5], n_filters[filter_idx-1], 1, 1, bias=False),
                                              nn.ReLU())
            self.gate0 = torch.nn.Sequential(nn.Conv2d(n_skips_radar[-5]+self.params.text_hidden_dim, n_filters[filter_idx-1], 1, 1, bias=False),
                                                 nn.Sigmoid())
            in_channels=n_filters[filter_idx-1]
            skip_channels=0
        else:
            print('Not supported.')


        self.deconv0 = DecoderBlock(
        in_channels=in_channels,
        skip_channels=skip_channels,
        out_channels=n_filters[filter_idx],
        activation_func=activation_func
        )

        self.get_depth = torch.nn.Sequential(nn.Conv2d(n_filters[filter_idx], 1, 3, 1, 1, bias=False),
                                         nn.Sigmoid())
        
        self.text_img_genatt = GenAttNew(params, n_skips_image[-1])
        self.text_img_regionatt = RegionAttNew(params, n_filters[0])
        self.conv = torch.nn.Sequential(nn.Conv2d(n_filters[0], n_filters[0], 3, 1, 1, bias=False),
                                                    nn.ReLU(inplace=True))


    def forward(self, skip_image, text_feat_gen, text_feat, text_mask, skip_radar, class_feat=None, uncertainty=None, shape=None):

        if 'swin' in self.params.encoder:
            for i, f in enumerate(skip_image):
                skip_image[i] = torch.permute(skip_image[i], (0, 3, 1, 2))
        

        if self.params.fuse == 'concat':
            skip_4 = torch.cat((skip_image[-1], skip_radar[-1]), dim=1)
            skip_3 = torch.cat((skip_image[-2], skip_radar[-2]), dim=1)
            deconv_4 = self.deconv4(skip_4, skip_3)
            deconv_4 = self.aspp(deconv_4)
            deconv_4 = self.daspp_conv(deconv_4)
            deconv_4 = deconv_4 + self.text_img_regionatt(deconv_4, text_feat, text_mask, scale_factor=1/16)
            deconv_4 = self.conv(deconv_4)

            skip_2 = torch.cat((skip_image[-3], skip_radar[-3]), dim=1)
            deconv_3 = self.deconv3(deconv_4, skip_2)

            skip_1 = torch.cat((skip_image[-4], skip_radar[-4]), dim=1)
            deconv_2 = self.deconv2(deconv_3, skip_1)

            if len(skip_image) < 5:
                skip_0 = skip_radar[-5]
            else:
                skip_0 = torch.cat((skip_image[-5], skip_radar[-5]), dim=1)
            deconv_1 = self.deconv1(deconv_2, skip_0)

            deconv_0 = self.deconv0(deconv_1, shape=shape)
            
        elif self.params.fuse == 'gated_fuse':
            rad_weight4 = self.weight4(skip_radar[4])
            rad_project4 = self.project4(skip_radar[4])
            radar4 = rad_weight4*rad_project4
            skip_4 = skip_image[-1] + radar4
            skip_4 = skip_4 + self.text_img_genatt(skip_4, text_feat_gen)
            skip_3 = skip_image[-2]
            deconv_4 = self.deconv4(skip_4, skip_3)

            rad_weight3 = self.weight3(skip_radar[3])
            rad_project3 = self.project3(skip_radar[3])
            radar3 = rad_weight3*rad_project3
            deconv_4 = deconv_4 + radar3
            deconv_4 = self.aspp(deconv_4)
            deconv_4 = self.daspp_conv(deconv_4)
            deconv_4 = deconv_4 + self.text_img_regionatt(deconv_4, text_feat, text_mask, scale_factor=1/16)
            deconv_4 = self.conv(deconv_4)
            skip_2 = skip_image[-3]
            deconv_3 = self.deconv3(deconv_4, skip_2)

            rad_weight2 = self.weight2(skip_radar[2])
            rad_project2 = self.project2(skip_radar[2])
            radar2 = rad_weight2*rad_project2
            deconv_3 = deconv_3 + radar2
            skip_1 = skip_image[-4]
            deconv_2 = self.deconv2(deconv_3, skip_1)

            rad_weight1 = self.weight1(skip_radar[1])
            rad_project1 = self.project1(skip_radar[1])
            radar1 = rad_weight1*rad_project1
            deconv_2 = deconv_2 + radar1
            if len(skip_image) < 5:
                skip_0 = None
            else:
                skip_0 = skip_image[-5]
            deconv_1 = self.deconv1(deconv_2, skip_0)

            rad_weight0 = self.weight0(skip_radar[0])
            rad_project0 = self.project0(skip_radar[0])
            radar0 = rad_weight0*rad_project0
            deconv_1 = deconv_1 + radar0
            deconv_0 = self.deconv0(deconv_1, shape=shape)
        
        elif self.params.fuse == 'wafb':
            class_feat = class_feat.unsqueeze(-1).unsqueeze(-1)

            rad_weight4 = self.weight4(skip_radar[4])
            _, _, W, H = skip_radar[4].shape
            rad_gate4 = self.gate4(torch.cat((skip_radar[4], F.interpolate(class_feat, (W, H))), 1))
            rad_project4 = self.project4(skip_radar[4])
            radar4 = rad_weight4*rad_project4 + rad_gate4*rad_project4
            skip_4 = skip_image[-1] + radar4
            skip_4 = skip_4 + self.text_img_genatt(skip_4, text_feat_gen)
            skip_3 = skip_image[-2]
            deconv_4 = self.deconv4(skip_4, skip_3)

            _, _, W, H = skip_radar[3].shape
            rad_gate3 = self.gate3(torch.cat((skip_radar[3], F.interpolate(class_feat, (W, H))), 1))
            rad_weight3 = self.weight3(skip_radar[3])
            rad_project3 = self.project3(skip_radar[3])
            radar3 = rad_weight3*rad_project3 + rad_gate3*rad_project3
            deconv_4 = deconv_4 + radar3
            deconv_4 = self.aspp(deconv_4)
            deconv_4 = self.daspp_conv(deconv_4)
            deconv_4 = deconv_4 + self.text_img_regionatt(deconv_4, text_feat, text_mask, scale_factor=1/16)
            deconv_4 = self.conv(deconv_4)
            skip_2 = skip_image[-3]
            deconv_3 = self.deconv3(deconv_4, skip_2)

            _, _, W, H = skip_radar[2].shape
            rad_weight2 = self.weight2(skip_radar[2])
            rad_gate2 = self.gate2(torch.cat((skip_radar[2], F.interpolate(class_feat, (W, H))), 1))
            rad_project2 = self.project2(skip_radar[2])
            radar2 = rad_weight2*rad_project2 + rad_gate2*rad_project2
            deconv_3 = deconv_3 + radar2
            skip_1 = skip_image[-4]
            deconv_2 = self.deconv2(deconv_3, skip_1)

            _, _, W, H = skip_radar[1].shape
            rad_weight1 = self.weight1(skip_radar[1])
            rad_gate1 = self.gate1(torch.cat((skip_radar[1], F.interpolate(class_feat, (W, H))), 1))
            rad_project1 = self.project1(skip_radar[1])
            radar1 = rad_weight1*rad_project1 + rad_gate1*rad_project1
            deconv_2 = deconv_2 + radar1
            if len(skip_image) < 5:
                skip_0 = None
            else:
                skip_0 = skip_image[-5]
            deconv_1 = self.deconv1(deconv_2, skip_0)

            _, _, W, H = skip_radar[0].shape
            rad_gate0 = self.gate0(torch.cat((skip_radar[0], F.interpolate(class_feat, (W, H))), 1))
            rad_weight0 = self.weight0(skip_radar[0])
            rad_project0 = self.project0(skip_radar[0])
            radar0 = rad_weight0*rad_project0 + rad_gate0*rad_project0
            deconv_1 = deconv_1 + radar0
            deconv_0 = self.deconv0(deconv_1, shape=shape)
        else:
            print('not supported')

        final_depth = self.params.max_depth * self.get_depth(deconv_0)

        return final_depth


class RegionAttNew(nn.Module):
    def __init__(self,
                params,
                image_channels
                ):
        super(RegionAttNew, self).__init__()
        self.params = params
        self.hidden_dim = params.text_hidden_dim
        if params.text_fuse == 'attention':
            self.attention = nn.ModuleList(
                                    [MultiHeadAttentionModule(image_channels, self.hidden_dim, self.hidden_dim, 4),
                                    MultiHeadAttentionModule(image_channels, self.hidden_dim, self.hidden_dim, 4),
                                    MultiHeadAttentionModule(image_channels, self.hidden_dim, self.hidden_dim, 4),
                                    MultiHeadAttentionModule(image_channels, self.hidden_dim, self.hidden_dim, 4),
                                    ])

        elif params.text_fuse == 'cross_attention':
            # image is query
            self.attention_img = nn.ModuleList(
                                    [MultiHeadAttentionModule(image_channels, self.hidden_dim, self.hidden_dim, 4),
                                    MultiHeadAttentionModule(image_channels, self.hidden_dim, self.hidden_dim, 4),
                                    MultiHeadAttentionModule(image_channels, self.hidden_dim, self.hidden_dim, 4),
                                    MultiHeadAttentionModule(image_channels, self.hidden_dim, self.hidden_dim, 4),
                                    ])
            self.attention_text = nn.ModuleList(
                                    [MultiHeadAttentionModule(self.hidden_dim, image_channels, self.hidden_dim, 4),
                                    MultiHeadAttentionModule(self.hidden_dim, image_channels, self.hidden_dim, 4),
                                    MultiHeadAttentionModule(self.hidden_dim, image_channels, self.hidden_dim, 4),
                                    MultiHeadAttentionModule(self.hidden_dim, image_channels, self.hidden_dim, 4),
                                    ])
            self.conv = torch.nn.Sequential(nn.Conv2d(2*self.hidden_dim, image_channels, 1, 1, bias=False),
                                              nn.ReLU())



    def forward(self, image_feature, text_feat, text_mask, scale_factor=1/16):

        text_mask_downsample = F.interpolate(text_mask, scale_factor=scale_factor, mode='nearest').long()
        B, C, H, W = image_feature.shape
        if self.params.text_fuse == 'attention':
            image_feat_res = []
            for b in range(B):
                # for each sample do it separately
                unique_number = torch.unique(text_mask_downsample[b], sorted=True)
                feat_res = []
                for idx in unique_number:
                    z = text_feat[idx-1][b:(b+1)].unsqueeze(1) # (1,1,C)
                    
                    image_feat = image_feature[b:b+1][:, :, text_mask_downsample[b, 0]==idx]
                
                    image_feat = self.attention[idx-1](image_feat.view(1, C, -1).permute((0, 2, 1)), z)

                    image_feat = image_feat.permute((0, 2, 1)).view((1, C, H, -1))

                    feat_res.append(image_feat)

                feat_res = torch.cat(feat_res, -1)

                image_feat_res.append(feat_res)
            image_feat_res = torch.cat(image_feat_res, 0)

        
        elif self.params.text_fuse == 'cross_attention':
            image_feat_res = []
            for b in range(B):
                # for each sample do it separately
                unique_number = torch.unique(text_mask_downsample[b], sorted=True)

                feat_res_img = []
                feat_res_text = []

                for idx in unique_number:
                    z = text_feat[idx-1][b:(b+1)].unsqueeze(1) # (1,1,C)
                    
                    image_feat = image_feature[b:b+1][:, :, text_mask_downsample[b, 0]==idx]            
                    
                    image_feat_att = self.attention_img[idx-1](image_feat.view(1, C, -1).permute((0, 2, 1)), z)
                    image_feat_att = image_feat_att.permute((0, 2, 1)).view((1, self.hidden_dim, H, -1))
                    feat_res_img.append(image_feat_att)

                    text_feat_att = self.attention_text[idx-1](z.repeat(1, image_feat.shape[-1], 1), image_feat.view(1, C, -1).permute((0, 2, 1)))
                    text_feat_att = text_feat_att.permute((0, 2, 1)).view((1, self.hidden_dim, H, -1))
                    feat_res_text.append(text_feat_att)

                feat_res_img = torch.cat(feat_res_img, -1)
                feat_res_text = torch.cat(feat_res_text, -1)

                feat_res = torch.cat((feat_res_img, feat_res_text), 1)
                
                image_feat_res.append(feat_res)
            image_feat_res = torch.cat(image_feat_res, 0)

            image_feat_res = self.conv(image_feat_res)

        else:
            print('Not supported yet.')

        return image_feat_res

class GenAttNew(nn.Module):
    def __init__(self,
                params,
                image_channels
                ):
        super(GenAttNew, self).__init__()
        self.params = params
        self.hidden_dim = params.text_hidden_dim
        if params.text_fuse == 'attention':
            self.attention_gen = MultiHeadAttentionModule(
                image_channels, image_channels, image_channels, 4
            )
        elif params.text_fuse == 'cross_attention':

            self.attention_gen_img = MultiHeadAttentionModule(
                image_channels, self.hidden_dim, self.hidden_dim, 4
            )
            self.attention_gen_text = MultiHeadAttentionModule(
                self.hidden_dim, image_channels, self.hidden_dim, 4,
            )
            self.conv_gen = torch.nn.Sequential(nn.Conv2d(2*self.hidden_dim, image_channels, 1, 1, bias=False),
                                              nn.ReLU())



    def forward(self, image_feature, text_feat_gen):

        B, C, H, W = image_feature.shape
        if self.params.text_fuse == 'attention':
            z = text_feat_gen.unsqueeze(1)
            image_feat = image_feature.view(B, C, -1).permute((0, 2, 1))
            # image_feat = self.attention(image_feat, z)
            feat_gen = self.attention_gen(image_feature, z)

            feat_gen = feat_gen.permute((0, 2, 1)).view((B, C, H, W))
            # image_feature = image_feature + image_feat
        
        elif self.params.text_fuse == 'cross_attention':
            z = text_feat_gen.unsqueeze(1)
            image_feat = image_feature.view(B, C, -1).permute((0, 2, 1))
            # image_feat = self.attention(image_feat, z)
            image_feat_gen = self.attention_gen_img(image_feat, z)

            text_feat_gen = self.attention_gen_text(z.repeat(1, image_feat.shape[1], 1), image_feat)

            feat_gen = torch.cat((image_feat_gen.permute((0, 2, 1)).view((B, self.hidden_dim, H, W)), text_feat_gen.permute((0, 2, 1)).view((B, self.hidden_dim, H, W))), 1)
            feat_gen = self.conv_gen(feat_gen)

        else:
            z = F.interpolate(text_feat_gen.unsqueeze(-1).unsqueeze(-1), size=(H, W))
            image_feature = image_feature + z

        return feat_gen
