# inference.py
import os
import torch
import torchaudio
from tqdm import tqdm
import argparse
from model import SecoustiCodec
from params import params
from hifigan.wav2mel import AudioProcessor  
from hifigan.mel2wav import infer_tts as hifi_infer 

class SecoustiCodecInference:
    def __init__(self, model_path, hifi_path, model_version='21.5', device=None):
        """
        初始化编解码器推理模型
        
        参数:
            model_path: 模型权重路径
            hifi_path: HiFi-GAN模型路径
            model_version: 模型版本 ('21.5' 或 '86')
            device: 计算设备 (默认: 自动选择GPU或CPU)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hifi_path = hifi_path
        
        # 根据版本设置ratios参数
        if model_version == '21.5':
            params._seanet_kwargs["ratios"] = [2, 2]
        elif model_version == '86':
            params._seanet_kwargs["ratios"] = [1, 1]
        else:
            raise ValueError(f"不支持的模型版本: {model_version}. 请选择 '21.5' 或 '86'")
        
        # 初始化模型
        self.model = SecoustiCodec(params).eval().to(self.device)
        self.processor = AudioProcessor()
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        
        print(f"已加载 {model_version}Hz 模型，设备: {self.device}")
    
    def _load_audio(self, wav_path):
        """加载音频并转换为mel特征"""
        _, feat, _ = self.processor.audio_to_mel(wav_path)
        return feat.to(self.device)  # 增加batch维度
    
    def encode(self, audio_path):
        """
        编码音频文件
        返回:
            acoustic_embed: 声学嵌入 [B, C, T]
            token: 离散token
            paralinguistic_embed: 副语言嵌入 [B, 512]
        """
        audio_feat = self._load_audio(audio_path)
        with torch.no_grad():
            acoustic_embed, token = self.model.encode_token(audio_feat)
            paralinguistic_embed = self.model.extract_prompt(audio_feat)
        return acoustic_embed, token, paralinguistic_embed
    
    def decode_from_token(self, token, paralinguistic_embed):
        """从语义token和副语言嵌入解码为音频波形"""
        with torch.no_grad():
            mel = self.model.decode_token(token, paralinguistic_embed)
        wav = hifi_infer(mel, self.hifi_path)
        return wav
    
    def decode_from_acoustic(self, acoustic_embed):
        """从声学嵌入解码为音频波形"""
        with torch.no_grad():
            mel = self.model.decode_acoustic(acoustic_embed)
        wav = hifi_infer(mel, self.hifi_path)
        return wav
    
    def process_audio(self, source_path, output_dir):
        """
        完整处理流程:
        1. 编码源音频
        2. 从提示音频提取副语言嵌入
        3. 解码两种版本
        4. 保存结果
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(source_path).split('.')[0]
        
        # 编码源音频
        acoustic_embed, token, paralinguistic_embed = self.encode(source_path)

        # 解码声学版本
        acoustic_wav = self.decode_from_acoustic(acoustic_embed)
        torchaudio.save(
            os.path.join(output_dir, f"{base_name}_acoustic.wav"),
            acoustic_wav.cpu(),
            sample_rate=params.sample_rate
        )
        
        # 解码语义token版本
        token_wav = self.decode_from_token(token, paralinguistic_embed)
        torchaudio.save(
            os.path.join(output_dir, f"{base_name}_semantic.wav"),
            token_wav.cpu(),
            sample_rate=params.sample_rate
        )

def main():
    parser = argparse.ArgumentParser(description='SecoustiCodec 推理')
    parser.add_argument('--source', type=str, required=True, help='源音频路径')
    parser.add_argument('--output_dir', type=str, default='./test', help='输出目录')
    parser.add_argument('--model_version', type=str, choices=['21.5', '86'], default='21.5',
                       help='模型版本 (21.5hz 或 86hz)')
    parser.add_argument('--model_path', type=str, default='./pretrain_models/secousticodec_21_5_hz.pt',
                       help='模型权重路径')
    parser.add_argument('--hifi_path', type=str, default='./pretrain_models/hift.pt',
                       help='HiFi-GAN 模型路径')
    
    args = parser.parse_args()
    
    # 初始化推理器
    inference = SecoustiCodecInference(
        model_path=args.model_path,
        hifi_path=args.hifi_path,
        model_version=args.model_version
    )
    
    # 处理音频
    inference.process_audio(
        source_path=args.source,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
'''
python inference.py \
  --source ./test/test.wav \
  --output_dir ./test \
  --model_version 86 \
  --model_path ./pretrain_models/secousticodec_86_hz.pt \
  --hifi_path ./pretrain_models/hift.pt


python inference.py \
  --source ./test/test.wav \
  --output_dir ./test \
  --model_version 21.5 \
  --model_path ./pretrain_models/secousticodec_21_5_hz.pt \
  --hifi_path ./pretrain_models/hift.pt
'''