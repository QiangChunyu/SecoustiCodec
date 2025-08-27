# SecoustiCodec: Cross-Modal Aligned Streaming Speech Codec

> Ultra-low bitrate speech codec (0.27-1 kbps) with cross-modal alignment and real-time capabilities

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2508.02849-b31b1b)](https://arxiv.org/abs/2508.02849)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/qiangchunyu/SecoustiCodec)
[![Demo](https://img.shields.io/badge/🚀-Live_Demo-blue)](https://qiangchunyu.github.io/SecoustiCodec_Page/)

## Key Features ✨

- **Ultra-Low Bitrate** (0.27-1 kbps)
- **Cross-Modal Alignment** (text-speech synchronization)
- **Speaker Preservation** (semantic-paralinguistic disentanglement)
- **Real-Time Processing** (streaming architecture)
- **High Efficiency** (VAE+FSQ quantization)

## Quick Start 🚀

0. **System Requirements**
```bash
# Ubuntu/Debian
sudo apt install sox libsox-dev ffmpeg

# macOS (via Homebrew)
brew install sox ffmpeg
```

1. **Clone & Install:**
```bash
git clone https://github.com/QiangChunyu/SecoustiCodec.git
cd SecoustiCodec
conda create -n secousticodec python=3.10
conda activate secousticodec
pip install -r requirements.txt
```

2. **Download Models:**
```bash
pip install huggingface-hub
hf download qiangchunyu/SecoustiCodec --local-dir pretrain_models
```

3. **Run Inference:**
```bash
# For 86Hz model
python inference.py \
  --source ./test/test.wav \
  --output_dir ./test \
  --model_version 86 \
  --model_path ./pretrain_models/secousticodec_86_hz.pt \
  --hifi_path ./pretrain_models/hift.pt

# For 21.5Hz model
python inference.py \
  --source ./test/test.wav \
  --output_dir ./test \
  --model_version 21.5 \
  --model_path ./pretrain_models/secousticodec_21_5_hz.pt \
  --hifi_path ./pretrain_models/hift.pt
```

### Output Files
| File Pattern              | Description                                  |
|---------------------------|---------------------------------------------|
| `{source}_acoustic.wav`   | Reconstruction from acoustic embeddings     |
| `{source}_semantic.wav`   | Reconstruction from semantic tokens         |


## Model Comparison 🧪

| Model Version | Frame Rate | Bitrate |
|---------------|------------|---------|
| 21.5          | 21.5 Hz    | 0.27 kbps |
| 86            | 86 Hz      | 1.0 kbps |

## Architecture Overview 🏗️
![Model Architecture](https://qiangchunyu.github.io/SecoustiCodec_Page/model.png)




## Citation 📚
```bibtex
@article{qiang2025secousticodec,
  title={SecoustiCodec: Cross-Modal Aligned Streaming Single-Codecbook Speech Codec},
  author={Qiang, Chunyu and Wang, Haoyu and Gong, Cheng and Wang, Tianrui and Fu, Ruibo and Wang, Tao and Chen, Ruilong and Yi, Jiangyan and Wen, Zhengqi and Zhang, Chen and Wang, Longbiao and Dang, Jianwu and Tao, Jianhua},
  journal={arXiv preprint arXiv:2508.02849},
  year={2025}
}

@article{qiang2025vq,
  title={VQ-CTAP: Cross-Modal Fine-Grained Sequence Representation Learning for Speech Processing},
  author={Qiang, Chunyu and Geng, Wang and Zhao, Yi and Fu, Ruibo and Wang, Tao and Gong, Cheng and Wang, Tianrui and Liu, Qiuyu and Yi, Jiangyan and Wen, Zhengqi and Zhang, Chen and Che, Hao and Wang, Longbiao and Dang, Jianwu and Tao, Jianhua},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  year={2025},
  publisher={IEEE}
}
```


## Acknowledgments 🙏
- [HiFiGAN](https://github.com/jik876/hifi-gan) for waveform generation
- [MIMICodec](https://huggingface.co/kyutai/mimi) for implementation reference

## License
SecoustiCodec is released under the Apache License 2.0. See [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) for details.
