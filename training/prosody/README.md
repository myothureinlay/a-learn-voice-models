# Mandarin pitch and syllable duration · 普通话基频与音节时长

These scripts adapt two projections in the A-Learn v2 Kokoro model. One predicts fundamental frequency (F0) over time. The other predicts how many frames each speech token occupies. Both use human Mandarin reading as supervision. The four voice embeddings retain their v2 speaker identities.

本目录提供两项微调的复现脚本。基频层预测语音音高随时间的变化，时长层预测发音符号占用的帧数。两项训练均以真人普通话朗读为依据，四个音色继续使用 v2 的说话人向量。

## Reproduce · 复现步骤

Prepare the pinned v2 model, speech data and pitch features first. The commands below continue in the same shell using the paths created during data preparation. The final command builds an ONNX file containing both updates and verifies its graph. [Data setup and hashes · 数据准备与哈希](DATA.md)

先按上述数据指南准备固定版本的 v2 模型、真人录音、对齐标注及基频特征。在同一个终端运行以下命令。最后一步将两组新参数写入同一 ONNX 文件，并验证模型结构。

```sh
out="$data/trained"
mkdir -p "$out"
"$py" "$scripts/fit_pitch.py" "$data/base/model.onnx" "$data/manifest.json" \
    "$data/features-v2" "$out"
"$py" "$scripts/fit_duration.py" "$data/base" "$data/align.tar.gz" "$out"
"$py" "$scripts/export_combined.py" "$data/base/model.onnx" "$out" "$out/combined.onnx"
```

The first command writes the pitch-only model to `trained/model.onnx`. The second command writes the duration parameters to `trained/duration-projection.npy`. The combined model from this run has SHA-256 `65551ecb7ab473632d7d4221165411a3853e32ed185c7df73d6f30959dd81584`. Native math libraries on different architectures can introduce small differences during fitting.

第一条命令输出单独调整基频的模型，第二条输出时长参数。第三条命令生成包含两项调整的 `trained/combined.onnx`。跨平台重训时，底层数值库可能产生细小的浮点差异。

## How the models were fitted · 训练方法

**Pitch.** For each aligned syllable, up to twelve voiced samples pair a model activation with a human pitch measurement at the same relative position. The training target uses the median pitch ratio within each utterance to match registers. Ridge regression fits the pitch values and their local changes. An additional penalty limits pitch changes in low-frequency frames. Validation on two separate speakers selected a 0.5 update from six strengths.

**基频。** 在每个对齐音节内部，按相对位置将模型特征与真人录音的有声基频配对，每个音节最多十二个位置。训练先根据句内基频比值的中位数对齐音域，再用岭回归拟合音高及相邻位置的变化，同时约束低基频帧。依据两名独立验证说话人的数据，从六档更新强度中选出 0.5。

**Duration.** The second fit starts from the v2 duration projection and reuses its speaker-separated 800/180/300-utterance split. Its target blends absolute human syllable lengths with lengths adjusted for the sentence's speaking rate. Training limits each target adjustment to 180 ms. Validation selects the strength using the overall error and constraints on very short syllables, sustained syllables and sentence length. This run selected full strength.

**时长。** 以 v2 的时长层为起点，沿用按说话人划分的 800 条训练句、180 条验证句和 300 条测试句。训练目标结合真人音节的原始时长和考虑整句语速后的时长，单个音节的目标调整幅度上限为 180 毫秒。验证时综合全体误差、短音节、长音节和整句长度，最终选出全量更新。

## Held-out checks · 留出集结果

| Metric · 平均绝对误差 | v2 | Adapted projection · 微调后 |
|---|---:|---:|
| Register-adjusted pitch, semitones · 音域对齐后基频，半音 | 3.478 | 2.781 |
| Local pitch change, semitones · 局部音高变化，半音 | 1.214 | 0.938 |
| Predicted syllable duration, ms · 音节预测时长，毫秒 | 85.82 | 82.38 |
| Human duration ≤150 ms · 短音节，毫秒 | 52.48 | 52.39 |
| Human duration ≥300 ms · 长音节，毫秒 | 259.11 | 243.36 |

Pitch uses 3,562 matched observations in 33 recordings from two held-out speakers. Duration uses 3,328 aligned syllables, including 339 short and 485 sustained syllables. Each row evaluates a single projection with controlled phonetic input. The duration windows can include low-energy and unvoiced frames. End-to-end pronunciation, naturalness and pause placement need separate tests of rendered speech. [Pitch results · 基频结果](pitch-results.json) · [Duration results · 时长结果](duration-results.json)

基频测试包括两名留出说话人的 33 条录音、3,562 个配对观测。时长测试包括 3,328 个对齐音节，其中 339 个短音节、485 个长音节。表内各项均在控制发音输入的条件下分别评估。音节预测窗口可含低能量或清音帧，完整合成语音的字音、自然度及停顿位置还需通过独立声学测试与听评检验。

## Rendered speech · 合成语音复核

The combined model is experimental. In 80 isolated syllable comparisons, mean pitch-shape error decreased from 2.062 to 1.963 semitones. Fourth-tone error increased from 3.090 to 3.437 semitones, and absolute audible-span error increased from 274.9 to 299.7 ms. Four voices read twenty tone examples, each with one reference recording. In the independent phrase set, duration error rose from 168.8 to 175.6 ms for eight short sentences and from 605.0 to 629.4 ms for eight long sentences. These measurements use the same original v2 lexicon, speed of 0.9 and silence scale of 0.2. [Complete comparison · 完整评测](evaluation/comparative-results.json)

合并模型为实验版本。四个音色分别朗读二十个带调单字，合成音与每个字的一条真人参考录音对比。平均音高曲线误差从 2.062 降至 1.963 半音，但第四声误差从 3.090 升至 3.437 半音，单字可听时长误差从 274.9 升至 299.7 毫秒。另测八个短句和八个长句，句长误差分别由 168.8 升至 175.6 毫秒、由 605.0 升至 629.4 毫秒。三组模型使用同一份 v2 原始词典、0.9 倍语速与 0.2 停顿系数。

To repeat the waveform checks, run the following after fitting. Install ffmpeg on Linux or use afconvert on macOS to decode the reference MP3s. The tone references are pinned to a revision in the source repository. [Reference recordings · 参考录音](https://github.com/davinfifield/mp3-chinese-pinyin-sound/tree/aa25ecce7b7fb02757b2c2b8e3c01aa975812edc)

训练完成后，可用以下命令复现波形对比。Linux 需安装 ffmpeg，macOS 可使用系统自带的 afconvert 解码参考录音。

```sh
"$py" -m pip install sherpa-onnx==2.0.1
evalscripts="$scripts/evaluation"
runs="$data/eval"
mkdir -p "$runs"
"$py" "$evalscripts/fetch_reference.py"
"$py" "$evalscripts/evaluate_waveforms.py" reference "$evalscripts/reference" "$runs/ref"
"$py" "$evalscripts/check_phrases.py" select "$data" "$runs/cases.json"
for run in v2 pitch combined; do
    case "$run" in
        v2) model="$data/base/model.onnx" ;;
        pitch) model="$out/model.onnx" ;;
        combined) model="$out/combined.onnx" ;;
    esac
    "$py" "$evalscripts/evaluate_waveforms.py" synth "$data/base" "$model" "$runs/$run"
    "$py" "$evalscripts/evaluate_waveforms.py" compare "$runs/ref/metrics.json" "$runs/$run/metrics.json" "$runs/$run-summary.json"
    "$py" "$evalscripts/check_phrases.py" measure "$data/base" "$model" "$runs/cases.json" "$runs/phrase-$run.json"
done
"$py" "$evalscripts/compare_runs.py" "$runs" "$runs/comparative-results.json"
```

## Credits · 来源和许可

The v2 pack contains the upstream Kokoro model and voice embeddings, the sherpa-onnx conversion, the Mandarin lexicon and normalization assets. AISHELL-3 is released by Beijing Shell Shell Technology Co., Ltd. under Apache 2.0. PaddleSpeech supplies the time alignments. The pitch measurements use Praat through Parselmouth. The reproduction environment installs those analysis tools separately. [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) · [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) · [AISHELL-3](https://www.openslr.org/93/) · [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/aishell3/tts3) · [Parselmouth](https://github.com/YannickJadoul/Parselmouth)

基础模型、语音向量、转换工具、词典和文本规范化文件的上游来源及许可见仓库声明。微调使用北京希尔贝壳科技有限公司发布的 AISHELL-3 普通话朗读语料与 PaddleSpeech 的对齐标注，基频由 Praat 和 Parselmouth 提取。[Notices · 来源声明](../../NOTICE.txt) · [Apache 2.0](../../LICENSE) · [pypinyin MIT](../../PYPINYIN-LICENSE)
