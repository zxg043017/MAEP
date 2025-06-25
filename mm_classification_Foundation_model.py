import torch
import torch.nn as nn
import torch.nn.functional as F

# from model.attention.DANet import DAModule,DASModule
from model.CLS.resnet import Deep_Vision_Feature_Model
from model.CLS.transformer_decoder import TransformerDecoder,TransformerDecoderLayer
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D
from einops import rearrange, repeat, reduce

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)

class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out


class MultimodalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultimodalCrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x1_feature, x2_features):
        # 调整形状为 [batch_size, num_patches, embed_dim]
        batch_size, channels= x1_feature.shape
        x1_features_flat = x1_feature.view(batch_size, channels, -1).transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        x2_features_flat = x2_features.view(batch_size, channels, -1).transpose(1, 2)  # [batch_size, num_patches, embed_dim]

        # Cross Attention
        fused_features, _ = self.attention(query=x1_features_flat, key=x2_features_flat, value=x2_features_flat)
        # fused_features = fused_features.transpose(1, 2).view(batch_size, channels, depth, height, width)  # 恢复形状
        fused_features = fused_features.transpose(1, 2).view(batch_size, channels)  # 恢复形状
        return fused_features, fused_features

class SingleAttention(nn.Module):
    def __init__(self, vis_dim=512) -> None:
        super(SingleAttention, self).__init__()

        self.vis_dim = vis_dim
        self.decoder_layer_global = TransformerDecoderLayer(d_model=self.vis_dim, nhead=8, normalize_before=True)
        self.decoder_norm_global = nn.LayerNorm(self.vis_dim)
        self.transformer_decoder_global = TransformerDecoder(decoder_layer=self.decoder_layer_global, num_layers=6,
                                                             norm=self.decoder_norm_global)

    def global_query_local_key_value(self, local_feature, global_feature):
        # global as queries
        # global output [2, 512] = > [2, 1, 512]   local output [2, 512] => [2, 512, 1]

        B = global_feature.shape[0]
        global_feature = torch.reshape(global_feature, (B, -1, self.vis_dim))
        local_feature = torch.reshape(local_feature, (B, self.vis_dim, 1))

        pos_embedding = PositionalEncoding1D(self.vis_dim)(torch.zeros(1, 1, self.vis_dim))  # b h/p w/p d/p dim
        pos_embedding = rearrange(pos_embedding, 'b h c -> h b c')  # n b dim

        pos = pos_embedding.to(local_feature.device)  # (H/P W/P D/P) B Dim
        image_embedding = rearrange(local_feature, 'b dim h -> h b dim')  # (H/P W/P D/P) B Dim
        queries = rearrange(global_feature, 'b n dim -> n b dim')  # N B Dim

        global_fused, _ = self.transformer_decoder_global(queries, image_embedding, pos=pos)  # N B Dim
        global_fused = rearrange(global_fused, 'n b dim -> (b n) dim')  # (B N) Dim

        return global_fused

    def forward(self, local_feature, global_feature):
        global_fused = self.global_query_local_key_value(local_feature, global_feature)
        align_feature = global_feature.clone()
        return global_fused, align_feature

class MultimodalFusionAttention(nn.Module):
    def __init__(self, channels):
        super(MultimodalFusionAttention, self).__init__()
        self.attention1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.attention2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, modality1_output, modality2_output):
        # 计算注意力权重
        weight1 = self.attention1(modality1_output)
        weight2 = self.attention2(modality2_output)

        # 加权融合
        fused_features = weight1 * modality1_output + weight2 * modality2_output
        return fused_features


class DoubleAttention(nn.Module):
    def __init__(self, vis_dim=256) -> None:
        super(DoubleAttention, self).__init__()

        self.vis_dim = vis_dim

        self.decoder_layer_vision = TransformerDecoderLayer(d_model=self.vis_dim, nhead=8, normalize_before=True)
        self.decoder_norm_vision = nn.LayerNorm(self.vis_dim)
        self.transformer_decoder_vision = TransformerDecoder(decoder_layer=self.decoder_layer_vision, num_layers=6, norm=self.decoder_norm_vision)
        self.decoder_layer_vl = TransformerDecoderLayer(d_model=self.vis_dim, nhead=8, normalize_before=True)
        self.decoder_norm_vl = nn.LayerNorm(self.vis_dim)
        self.transformer_decoder_vl = TransformerDecoder(decoder_layer=self.decoder_layer_vl, num_layers=6, norm=self.decoder_norm_vl)

    def vision_query_vl_key_value(self, vl_feature, vision_feature):
        # vision as queries
        # vision_feature output [2, 512] = > [2, 1, 512]   vl_feature output [2, 512] => [2, 512, 1]
        B = vision_feature.shape[0]
        vision_feature = torch.reshape(vision_feature, (B, -1, self.vis_dim))
        vl_feature = torch.reshape(vl_feature, (B, self.vis_dim, 1))
        pos_embedding = PositionalEncoding1D(self.vis_dim)(torch.zeros(1, 1, self.vis_dim))  # b h/p w/p d/p dim
        pos_embedding = rearrange(pos_embedding, 'b h c -> h b c')  # n b dim
        pos = pos_embedding.to(vl_feature.device)  # (H/P W/P D/P) B Dim
        image_embedding = rearrange(vl_feature, 'b dim h -> h b dim')  # (H/P W/P D/P) B Dim
        queries = rearrange(vision_feature, 'b n dim -> n b dim')  # N B Dim
        vision_fused, _ = self.transformer_decoder_vision(queries, image_embedding, pos=pos)  # N B Dim
        vision_fused = rearrange(vision_fused, 'n b dim -> (b n) dim')  # (B N) Dim

        return vision_fused

    def vl_query_vision_key_value(self, vl_feature, vision_feature):
        ## vsion-language (vl) as queries
        ## vl output [2, 512] => [2, 1, 512]   vision_feature output [2 512] => [2, 512, 1]
        B = vl_feature.shape[0]
        vl_feature = torch.reshape(vl_feature, (B, -1, self.vis_dim))
        vision_feature = torch.reshape(vision_feature, (B, self.vis_dim, 1))
        pos_embedding = PositionalEncoding1D(self.vis_dim)(torch.zeros(1, 1, self.vis_dim))  # b h/p w/p d/p dim
        pos_embedding = rearrange(pos_embedding, 'b h c -> h b c')  # n b dim
        pos = pos_embedding.to(vision_feature.device)  # (H/P W/P D/P) B Dim
        image_embedding = rearrange(vision_feature, 'b dim h -> h b dim')  # (H/P W/P D/P) B Dim
        queries = rearrange(vl_feature, 'b n dim -> n b dim')  # N B Dim
        vl_fused, _ = self.transformer_decoder_vl(queries, image_embedding, pos=pos)  # N B Dim
        vl_fused = rearrange(vl_fused, 'n b dim -> (b n) dim')  # (B N) Dim

        return vl_fused

    def forward(self, vl_feature, vision_feature):
        vision_fused = self.vision_query_vl_key_value(vl_feature, vision_feature)
        vl_fused = self.vl_query_vision_key_value(vl_feature, vision_feature)
        fusion_feature = torch.cat((vision_fused, vl_fused), dim=1)
        # vl_align_feature = vision_feature + vl_feature
        vl_align_feature = torch.cat((vision_feature, vl_feature), dim=1)
        return fusion_feature, vl_align_feature
class UnetEncoder(nn.Module):
    def __init__(self, act='relu'):
        super(UnetEncoder, self).__init__()

        self.down_tr64 = DownTransition(1,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)

        return self.out512

class FeatureFusionConcat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionConcat, self).__init__()
        # 用卷积层对拼接后的特征进行处理
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.controller = nn.Linear(512, 256)

    def forward(self, x1_features, x2_features):
        # 拼接两个模态的特征 (在通道维度)
        fused_features = torch.cat([x1_features, x2_features], dim=1)  # [batch_size, in_channels, depth, height, width]
        # 卷积处理
        fused_features = self.controller(fused_features)
        return fused_features, fused_features

class Vision_Language_Merge_Branch(nn.Module):
    def __init__(self, out_channels=3, text_prompt=True, encoding="word_embedding", text_embedding_name="CLIP_embedding") -> None:  ## change to 3
        super(Vision_Language_Merge_Branch, self).__init__()
        self.text_prompt = text_prompt
        self.encoding = encoding
        self.text_embedding_name = text_embedding_name
        self.encoder = UnetEncoder()
        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.Flatten()
        )
        self.cls_head = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=out_channels)
        )
        self.controller = nn.Linear(128, 256)
        if self.encoding == 'rand_embedding':
            self.organ_embedding = nn.Embedding(out_channels, 512)
        elif self.encoding == 'word_embedding':
            self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
            if self.text_embedding_name =="CLIP_embedding":
                self.text_to_vision = nn.Linear(512, 128)
            elif self.text_embedding_name =="Bert_embedding":
                self.text_to_vision = nn.Linear(768, 128)


        self.class_num = out_channels

    def load_params(self, model_dict):
        store_dict = self.encoder.state_dict()
        for key in model_dict.keys():
            if "down_tr" in key:
                store_dict[key.replace("module.backbone.", "")] = model_dict[key]
        self.encoder.load_state_dict(store_dict)

        print('Use pretrained weights')

    def forward(self, x1, x2):
        x1_feature = self.encoder(x1)
        x1_feature = self.GAP(x1_feature)

        x2_feature = self.encoder(x2)
        x2_feature = self.GAP(x2_feature)

        B = x1_feature.shape[0]

        x1_feature = x1_feature.unsqueeze(1)
        x2_feature = x2_feature.unsqueeze(1)

        fusion_feature = torch.cat([x1_feature, x2_feature], 1)

        if self.text_prompt:
            print("use text prompt!")
            # text embedding of modality x1
            x1_text_embedding = F.relu(self.text_to_vision(self.organ_embedding[0]))
            x1_text_embedding = x1_text_embedding.unsqueeze(0).repeat(B, 1, 1)
            x1_VL_feature = torch.mul(x1_feature, x1_text_embedding)

            # text embedding of modality x2
            x2_text_embedding = F.relu(self.text_to_vision(self.organ_embedding[1]))
            x2_task_encoding = x2_text_embedding.unsqueeze(0).repeat(B, 1, 1)
            x2_VL_feature = torch.mul(x2_feature, x2_task_encoding)
            vl_fusion_feature = torch.cat([x1_VL_feature, x2_VL_feature], 1)
            vl_fusion_feature = self.controller(vl_fusion_feature).mean(dim=1)
            feature = vl_fusion_feature
        else:
            print("not use text prompt!")

            feature = fusion_feature.view(B, 256)
        return feature

class Foundation_Model_Classification(nn.Module):
    def __init__(self, n_class = 7, text_prompt = True, VL_CrossAttention = True, fusion_module="Cross_Attention", text_prompt_name="CLIP_embedding", text_encoding="word_embedding", res_depth=50) -> None:
        super(Foundation_Model_Classification, self).__init__()
        self.text_prompt = text_prompt
        self.text_encoding = text_encoding
        self.res_depth = res_depth
        self.text_prompt_name = text_prompt_name
        self.VL_CrossAttention = VL_CrossAttention
        self.VLM_branch = Vision_Language_Merge_Branch(n_class, text_prompt, encoding=text_encoding, text_embedding_name=text_prompt_name)
        self.DVF_branch = Deep_Vision_Feature_Model(model_depth=res_depth, n_classes=256, input_W=96, input_H=96, input_D=96)
        self.VL_Fusion_module = DoubleAttention(vis_dim=256)
        self.SingleFusionModule = SingleAttention(vis_dim=256)

        self.cls_head_crossfusion = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_class)
        )
        self.cls_head_singlefusion = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_class)
        )

        if fusion_module=="DoubleAttention":
            self.multimoal_fusion_module = DoubleAttention(vis_dim=256)
        elif fusion_module=="Attention_Fusion":
            self.multimoal_fusion_module = MultimodalFusionAttention(channels=256)
        elif fusion_module=="Cross_Attention":
            self.multimoal_fusion_module = MultimodalCrossAttention(embed_dim=256, num_heads=2)
        elif fusion_module=="Concat_Fusion":
            self.multimoal_fusion_module = FeatureFusionConcat(in_channels=512, out_channels=256)
        elif fusion_module=="SingleAttention":
            self.multimoal_fusion_module = SingleAttention(vis_dim=256)

    def load_params(self, model_dict):
        self.VLM_branch.load_params(model_dict)

    def forward(self, x1, x2):
        VL_feature = self.VLM_branch(x1, x2)
        DVF_x1 = self.DVF_branch(x1)
        DVF_x2 = self.DVF_branch(x2)
        vision_feature_fusion, _ = self.multimoal_fusion_module(DVF_x1, DVF_x2)

        if self.VL_CrossAttention:
            fusionfeature, alignfeature = self.VL_Fusion_module(VL_feature, vision_feature_fusion)
            out = self.cls_head_crossfusion(fusionfeature)
            return out, alignfeature

        else:
            fusionfeature, alignfeature = self.SingleFusionModule(localfeature, globalfeature)
            out = self.cls_head_singlefusion(fusionfeature)

            return out, alignfeature

    if __name__ == "__main__":
        liver = torch.ones((2, 1, 96, 96, 96))
        spleen = torch.ones((2, 1, 96, 96, 96))
        left_kidney = torch.ones((2, 1, 96, 96, 96))
        right_kidney = torch.ones((2, 1, 96, 96, 96))
        model = UnetClassification(n_class=4)
        # load pretrain model
        pretrain = "/home/zxg/zxg_code/TransUnet_Challenge/unet_classification/pretrain_model/unet.pth"
        model.load_params(torch.load(pretrain, map_location='cpu')['net'])
        # pred = model(liver, spleen, left_kidney, right_kidney)
        pred = model(liver)
        print(pred)