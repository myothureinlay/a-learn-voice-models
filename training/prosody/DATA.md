# Mandarin pitch adaptation data · 普通话基频微调数据

AISHELL-3 provides human Mandarin recordings, speaker metadata and transcripts. The sampling script uses the publicly listed files from revision `f20d5db4a31fe779ef07bb1af4ea92da5c786622` of the AISHELL mirror, dated January 8, 2024. PaddleSpeech provides the tone-labelled alignments. The sampling date does not indicate a recording date. [AISHELL-3 and license](https://www.openslr.org/93/) · [Pinned corpus](https://huggingface.co/datasets/AISHELL/AISHELL-3/tree/f20d5db4a31fe779ef07bb1af4ea92da5c786622) · [Alignment source](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/tts3)

录音、转录和说话人资料来自 AISHELL-3。复现使用上列固定版本中可直接下载的录音和 PaddleSpeech 带声调的时间对齐。镜像版本日期为 2024 年 1 月 8 日。原始录音的采集日期未核实。

## Reproduce · 复现

Run from the repository root with Python 3.12, curl, unzip and GitHub CLI. The commands download around 80 MB of recordings and a 356 MB base pack. The reviewed shell commands below assume a POSIX environment. The reference data and model files can be stored outside the repository by changing `data`.

在仓库根目录运行，环境需具备 Python 3.12、curl、unzip 和 GitHub CLI。脚本会校验说话人资料、转录、对齐文件和音频的散列值。

```sh
root=$(pwd)
data="$root/prosody-data"
mkdir -p "$data/base"
python3.12 -m venv "$data/venv"
py="$data/venv/bin/python"
"$py" -m pip install -r training/requirements.txt praat-parselmouth==0.4.7
gh release download voicepack-v2 --repo myothureinlay/a-learn-voice-models \
  --pattern a-learn-kokoro-mandarin-v2.zip --dir "$data"
cd "$data"
echo '51c97e75a7bafa04e2698dd715931d330a2d3bda1a5e4c421ded60fa9ee73959  a-learn-kokoro-mandarin-v2.zip' | shasum -a 256 -c -
unzip -qo a-learn-kokoro-mandarin-v2.zip -d base
scripts="$root/training/prosody"
"$py" "$scripts/select_audio.py" prepare
"$py" "$scripts/select_audio.py" select
echo '9a44dc0c30b1543e293408673bf75eb0820e16672a9dec4b43b27697e109dace  manifest.json' | shasum -a 256 -c -
"$py" "$scripts/select_audio.py" fetch
(cd audio && shasum -a 256 -c "$scripts/reference-wav.sha256")
"$py" "$scripts/features.py" --base "$data/base" --data "$data"
```

## Coverage · 覆盖范围

| Split · 分组 | Speakers · 人数 | Recordings downloaded · 下载条数 | Used · 使用条数 | Syllables with paired pitch · 配对音节数 | Matched observations · 配对帧数 |
|---|---:|---:|---:|---:|---:|
| Train · 训练 | 17 | 225 | 201 | 1,783 | 18,432 |
| Validation · 验证 | 2 | 28 | 27 | 263 | 2,750 |
| Test · 测试 | 2 | 36 | 33 | 343 | 3,562 |

Test speakers are SSB0590 and SSB0609. Validation speakers are SSB0273 and SSB0710. The three groups have disjoint speaker IDs, transcript strings and tone-labelled syllable sequences. The 289 downloaded WAV files are mono 44.1 kHz, 16-bit PCM. Some recordings contain tone-labelled syllables missing in the model's lexicon lookup. Those recordings are excluded. Each matched observation pairs a model feature vector with a voiced Praat F0 estimate inside the corresponding aligned human syllable. The model runs at the app's default rate of 0.90. Pitch supervision uses up to twelve samples per syllable. Low-confidence and unvoiced frames are filtered.

测试说话人为 SSB0590 和 SSB0609，验证说话人为 SSB0273 和 SSB0710。训练、验证和测试组的说话人、文本及带调音节序列均无交叉。提取过程中排除含有模型发音表未收录音节的整条录音。在每个对齐音节内，将模型特征与真人音频中有效的基频估计配对，最多保留十二个位置。逐调覆盖情况可见 [feature-coverage.json](feature-coverage.json)。

The selection includes male read speech, six to 25 syllables per utterance and internal pauses up to 130 ms. The pitch test covers connected read speech from two held-out speakers. Isolated syllables and longer pauses require separate evaluation. Human listening tests and independent recordings complement this pitch test.

本次样本为男性朗读，单句含 6 至 25 个音节，句内停顿不超过 130 毫秒。对齐音节的音高测量需结合独立单字、含长停顿的句子和真人听评判断。
