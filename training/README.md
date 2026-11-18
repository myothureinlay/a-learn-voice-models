# Mandarin timing experiment · 普通话音节时长实验

We trained one low-rank update to Kokoro's final duration projection. Its 512 learned coefficients shift the duration-bin logits as a function of the existing recurrent features. The shift is folded into the 512 × 50 projection for Android. The four speaker embeddings, pitch predictor and waveform generator retain their upstream weights.

本次只微调 Kokoro 的时长预测输出层。512 个可训练系数根据已有的上下文特征调整时长分布，并合并到 512 × 50 的投影权重中。四种音色的说话人向量、基频预测器和声码器沿用原始权重。

## Data and split · 数据与划分

Targets are syllable intervals from PaddleSpeech's forced alignments of AISHELL-3 human Mandarin read speech. We use consecutive labeled syllables in short utterances, excluding internal silence and unusually short or long syllables. A speaker hash assigns the split. Identical Pinyin transcripts occur in at most one split. A deterministic hash assigns one of the four app voices to each utterance. The proxy input is reconstructed from the corpus Pinyin and the pack's Mandarin lexicon. It does not test the entire text-normalization pipeline.

目标来自 AISHELL-3 真人普通话朗读录音的 PaddleSpeech 强制对齐标注。筛选连续、有标注且时长合理的音节。按说话人划分训练、验证和测试集，剔除重复的拼音文本；每句以固定规则分配一种应用音色。模型输入由语料拼音和语音包词典重建。此项评估覆盖音节时长，完整的文本处理和主观自然度需另行评估。

| Split / 集合 | Speakers / 人数 | Utterances / 句数 | Syllables / 音节数 |
|---|---:|---:|---:|
| Training / 训练 | 93 | 800 | 8,427 |
| Validation / 验证 | 18 | 180 | 2,050 |
| Held-out test / 测试 | 32 | 300 | 3,328 |

The fit uses ridge regression on the linearized duration head. Validation speakers select the scale. Held-out normalized MAE compares each predicted syllable with its aligned reference after scaling the utterance to the reference speaking rate. It fell from 92.814 to 77.074 ms. Absolute MAE without that normalization fell from 98.830 to 85.817 ms at app speed 0.9. These are timing metrics; human listening scores have not been collected.

拟合使用带正则项的线性化时长回归，更新幅度由验证集选择。将每句预测的总时长缩放到参考语速后，测试集的音节平均绝对误差由 92.814 降至 77.074 毫秒。按应用默认语速 0.9 直接比较，误差由 98.830 降至 85.817 毫秒。尚未进行听众自然度评分。

## Reproduction · 复现

Use Python 3.12. Extract the original voicepack-v1 ZIP into `base/` and download the linked alignment archive. The script checks both input digests. It writes the full ONNX model, the compact projection, and a metrics report. Floating-point results can vary across libraries and CPU architectures.

使用 Python 3.12。把 v1 语音包解压到 `base/`，下载下方时间对齐压缩包。脚本核对输入摘要，并输出 ONNX 模型、时长层权重及评估报告。不同平台的浮点舍入可能略有差异。

```sh
python -m pip install -r training/requirements.txt
python training/train_timing.py base aishell3_alignment_tone.tar.gz output
```

- [AISHELL-3 · OpenSLR 93 · Apache 2.0](https://www.openslr.org/93/)
- [PaddleSpeech · AISHELL-3 alignment documentation / 时间对齐说明](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/tts3)
- [Alignment archive / 对齐数据](https://paddlespeech.cdn.bcebos.com/MFA/AISHELL-3/with_tone/aishell3_alignment_tone.tar.gz)
- [Metrics / 评估数据](timing-v2-metrics.json)
