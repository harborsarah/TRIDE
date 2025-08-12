import torch
import torch.nn as nn
from models.image_models import *
from models.text_models import *
from models.net_utils import *
from models.radar_models import *
import torch.nn.functional as F



class TRIDE(nn.Module):
    def __init__(self, params):
        super(TRIDE, self).__init__()
        self.params = params
        self.image_encoder = encoder(params)
        self.radar_encoder = encoder_radar_2d3d(params)
        self.text_encoder = TextEncoderSepPointEnrich(params, self.image_encoder)
        self.decoder = Unet_decoder(
                                    params, \
                                    params.n_filters_decoder,
                                    self.image_encoder.feat_out_channels,
                                    self.radar_encoder.feat_out_channels
                                    )
            
        self.hidden_dim = self.params.text_hidden_dim
    
    def forward(self, image, radar, radar_points, text_feature_general, text_feature_left, text_feature_mid_left, \
                text_feature_mid_right, text_feature_right, text_mask, text_length):
        
        image_output = self.image_encoder(image)
        radar_output, radar_point_feat = self.radar_encoder(radar, radar_points[:, :-1]) # last feature is to aligh with the text

        if self.params.use_img_feat:
            text_feat_gen, text_feat, class_pred, class_feat = self.text_encoder(text_feature_general, text_feature_left, text_feature_mid_left, \
                                                                                text_feature_mid_right, text_feature_right, text_length, radar_point_feat, radar_points[:, -1], image_output[-1])
        else:
            text_feat_gen, text_feat, class_pred, class_feat = self.text_encoder(text_feature_general, text_feature_left, text_feature_mid_left, \
                                                                                text_feature_mid_right, text_feature_right, text_length, radar_points[:, -1])
        final_depth = self.decoder(image_output, text_feat_gen, text_feat, text_mask, radar_output, class_feat)

        return final_depth, class_pred


