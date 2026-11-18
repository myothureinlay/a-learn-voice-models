# A-Learn Mandarin voices · A-Learn 普通话语音包

[Android and voice downloads / 安卓应用与语音包](https://github.com/myothureinlay/a-learn-voice-models/releases/tag/voicepack-v2) · [Version history / 版本记录](CHANGELOG.md) · [Training and evaluation / 训练与评估](training/README.md)

![v2 pack downloads / v2 语音包下载次数](https://img.shields.io/github/downloads/myothureinlay/a-learn-voice-models/voicepack-v2/a-learn-kokoro-mandarin-v2.zip?label=v2%20voice%20pack)
![v1 pack downloads / v1 语音包下载次数](https://img.shields.io/github/downloads/myothureinlay/a-learn-voice-models/voicepack-v1/a-learn-kokoro-mandarin-v1.zip?label=v1%20voice%20pack)

The A-Learn Android dictionary reads Mandarin on the device. Version 2 adapts the syllable timing of Kokoro's four existing voices. Download a voice pack once in Settings → Mandarin Voices. If v1 is installed, choose **Update voice rhythm** to update locally. Local updates do not download a second pack. GitHub counts asset requests, including retries. The badge may be cached.

A-Learn 安卓词典在设备上朗读普通话。第二版针对 Kokoro 四种现有音色微调了音节时长。首次使用时，在设置的普通话语音栏下载语音包。已安装 v1 时可点选 **Update voice rhythm** 本地更新。本地更新不会重复下载整包。GitHub 统计文件下载请求次数，其中包含重试，徽章数字可能有缓存。

## Listening comparison · 试听对比

Both clips use Yunxi, default speed, and the same text: 春天到了，我们一起去公园散步吧。

两段音频均为云希、默认语速，朗读相同的句子。 [Original / 原版](https://github.com/myothureinlay/a-learn-voice-models/releases/download/voicepack-v2/yunxi-before.wav) · [Trained timing / 微调后](https://github.com/myothureinlay/a-learn-voice-models/releases/download/voicepack-v2/yunxi-after.wav)

## Sources and licenses · 来源与许可

The base model and voice vectors are from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). The Android app uses [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Both projects declare Apache 2.0. A-Learn's timing update uses [AISHELL-3](https://www.openslr.org/93/) annotations from [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/tts3).

基础模型和音色向量来自 Kokoro-82M，安卓推理使用 sherpa-onnx。二者均标注 Apache 2.0。时长微调采用 AISHELL-3 真人朗读语料的 PaddleSpeech 对齐标注。词典、规范化数据及上游训练资料的来源和许可见 [NOTICE](NOTICE.txt)、[Apache 2.0](LICENSE) 和 [pypinyin MIT](PYPINYIN-LICENSE)。

The lexicon uses Misaki and pypinyin. The notice preserves upstream training-data acknowledgements and the limits of the normalization-data provenance. The pack includes the applicable notices.
