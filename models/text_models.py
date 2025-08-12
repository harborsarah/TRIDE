import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_utils import MultiHeadAttentionModule

class TextEncodeBlock(nn.Module):
    def __init__(self, params, image_encoder):
        super(TextEncodeBlock, self).__init__()
        num_channels = image_encoder.feat_out_channels[-1]
        self.num_text_feat = 768
        self.params = params
        self.hidden_dim = params.text_hidden_dim        

        self.feat = nn.Sequential(
        nn.Linear(self.num_text_feat, self.hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=2, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)


            
    def forward(self, text_emb, text_length):
        text_emb = self.feat(text_emb)

        text_emb_list = [] 
        for i in range(text_emb.shape[0]):
            text_emb_wo = text_emb[i:i+1, :int(text_length[i])]
            text_emb_wo, _ = self.lstm(text_emb_wo)

            # take the last feature OR do average?
            text_emb_wo = self.linear(text_emb_wo[:, -1, :]) # take the last feature 

            text_emb_list.append(text_emb_wo)
        
        text_emb_avg = torch.cat(text_emb_list, dim=0) # (B, hidden_dim)
        return text_emb_avg
        
        
class TextEncoderSepPointEnrich(nn.Module):
    def __init__(self, params, image_encoder):
        super(TextEncoderSepPointEnrich, self).__init__()
        num_class = 3
        self.params = params
        self.text_general_block = TextEncodeBlock(params, image_encoder)
        self.text_left_block = TextEncodeBlock(params, image_encoder)
        self.text_mid_left_block = TextEncodeBlock(params, image_encoder)
        self.text_mid_right_block = TextEncodeBlock(params, image_encoder)
        self.text_right_block = TextEncodeBlock(params, image_encoder)

        self.classifier = nn.Sequential(
            nn.Linear(params.text_hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_class),
            # nn.Softmax(dim=1)
        )
        self.use_img_feat = params.use_img_feat

        self.text_hidden_dim = params.text_hidden_dim
        self.point_hidden_dim = params.point_hidden_dim
        self.attention_text = nn.ModuleList(
                                    [MultiHeadAttentionModule(self.text_hidden_dim, self.point_hidden_dim, self.text_hidden_dim, 4),
                                    MultiHeadAttentionModule(self.text_hidden_dim, self.point_hidden_dim, self.text_hidden_dim, 4),
                                    MultiHeadAttentionModule(self.text_hidden_dim, self.point_hidden_dim, self.text_hidden_dim, 4),
                                    MultiHeadAttentionModule(self.text_hidden_dim, self.point_hidden_dim, self.text_hidden_dim, 4),
                                    ])
    
    def forward(self, text_feature_general, text_feature_left, text_feature_mid_left, \
                text_feature_mid_right, text_feature_right, text_length, radar_point_feat, radar_point_mask, image_last_feat=None):
        device = text_feature_general.device
        text_feature_general = self.text_general_block(text_feature_general, text_length[:, 0])
        text_feature_left = self.text_left_block(text_feature_left, text_length[:, 1])
        text_feature_mid_left = self.text_mid_left_block(text_feature_mid_left, text_length[:, 2])
        text_feature_mid_right = self.text_mid_right_block(text_feature_mid_right, text_length[:, 3])
        text_feature_right = self.text_right_block(text_feature_right, text_length[:, 4])
        text_feat = [text_feature_left, text_feature_mid_left, text_feature_mid_right, text_feature_right]
        # print(text_feature_right.shape)
        # radar_point_mask = radar_point_mask.unsqueeze(1)
        res_feat_left = []
        res_feat_mid_left = []
        res_feat_right = []
        res_feat_mid_right = []
        res_feat = [res_feat_left, res_feat_mid_left, res_feat_mid_right, res_feat_right]
        for b in range(text_feature_right.shape[0]):
            unique_number = torch.unique(radar_point_mask[b], sorted=True).int()
            for idx in range(1, 5):
                if idx in unique_number:
                    z = text_feat[idx-1][b:(b+1)].unsqueeze(1)
                    radar_point = radar_point_feat[b][radar_point_mask[b]==idx].unsqueeze(0)
                    z = self.attention_text[idx-1](z, radar_point).squeeze(1)
                    # res_feat.append(z)
                    res_feat[idx-1].append(z)
                else:
                    res_feat[idx-1].append(torch.zeros((1, self.text_hidden_dim), requires_grad=True, device=device))
        
        res_feat_left = torch.cat(res_feat_left, 0)
        res_feat_mid_left = torch.cat(res_feat_mid_left, 0)
        res_feat_mid_right = torch.cat(res_feat_mid_right, 0)
        res_feat_right = torch.cat(res_feat_right, 0)

        text_feature_left = text_feature_left + res_feat_left
        text_feature_mid_left = text_feature_mid_left + res_feat_mid_left
        text_feature_mid_right = text_feature_mid_right + res_feat_mid_right
        text_feature_right = text_feature_right + res_feat_right

        # if image_last_feat is not None:
        if self.use_img_feat:
            img_feat = torch.mean(image_last_feat, dim=(2, 3))
            img_feat = nn.functional.adaptive_avg_pool1d(img_feat, self.params.text_hidden_dim)
            classification_feat = text_feature_general + img_feat
            class_pred = self.classifier(classification_feat)

            # img_feat = self.img(torch.mean(image_last_feat, dim=(2, 3)))
            # class_pred = self.classifier(text_feature_general + img_feat)
        else:
            classification_feat = text_feature_general
            class_pred = self.classifier(classification_feat)

        return text_feature_general, [text_feature_left, text_feature_mid_left, text_feature_mid_right, text_feature_right], class_pred, classification_feat
