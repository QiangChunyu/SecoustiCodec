# model.py
import torch
from torch import nn
from modules import SEANetEncoder, SEANetDecoder, Semantic_Connector, SemanticTransformerVAE, Stransformer, ParalinguisticEncoder, FSQ

class SecoustiCodec(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.paralinguistic_encoder = ParalinguisticEncoder(params)
        self.fsq = FSQ(params.fsq_levels, 512)
        self.encoder = SEANetEncoder(**params._seanet_kwargs)
        self.decoder = SEANetDecoder(**params._seanet_kwargs)
        self.encoder_transformer = Stransformer.ProjectedTransformer(**params._encoder_transformer_kwargs)
        self.decoder_transformer = Stransformer.ProjectedTransformer(**params._decoder_transformer_kwargs)
        self.vae_semantic_transformer = SemanticTransformerVAE(params)
        self.semantic_connector = Semantic_Connector(params)
    
    def extract_prompt(self, prompt_feat):
        prompt_feat = prompt_feat[:, prompt_feat.shape[1]//2-self.params.prompt_feat_length//2:prompt_feat.shape[1]//2+self.params.prompt_feat_length//2,:]
        """提取副语言嵌入"""
        _, paralinguistic_embed, _ = self.paralinguistic_encoder(prompt_feat)
        return paralinguistic_embed

    def encode_token(self, feat):
        """
        编码音频为token和副语言嵌入
        返回:
            acoustic_embed: 声学嵌入 [B, C, T]
            token: 离散token (indices)
            paralinguistic_embed: 副语言嵌入 [B, 512]
        """
        feat = feat.transpose(1, 2)
        z = self.encoder(feat)
        
        acoustic_embed = self.encoder_transformer(z)[0]
        _, semantic_mu, _ = self.vae_semantic_transformer(z)
        _, token = self.fsq(semantic_mu)
        
        return acoustic_embed, token

    def decode_token(self, token, paralinguistic_embed):
        """
        从token和副语言嵌入解码为音频
        参数:
            token: 离散token (indices)
            paralinguistic_embed: 副语言嵌入 [B, 512]
        返回:
            mel: 重构的mel谱 [B, T, 80]
        """
        # 从token获取量化表示
        z_q = self.fsq.indices_to_codes(token)
        z_q = z_q.transpose(1, 2)
        
        # 连接嵌入
        connect_embed = self.semantic_connector(z_q, paralinguistic_embed)
        
        # 解码
        mel = self.decoder_transformer(connect_embed)[0]
        mel = self.decoder(mel)
        return mel.transpose(1, 2)

    def decode_acoustic(self, acoustic_embed):
        """
        直接从声学嵌入解码为音频
        参数:
            acoustic_embed: 声学嵌入 [B, C, T]
        返回:
            mel: 重构的mel谱 [B, T, 80]
        """
        mel = self.decoder_transformer(acoustic_embed)[0]
        mel = self.decoder(mel)
        return mel.transpose(1, 2)
