# A-Learn offline Mandarin voice pack

![Voice pack downloads](https://img.shields.io/github/downloads/myothureinlay/a-learn-voice-models/voicepack-v1/total?label=Voice%20pack%20downloads)

This repository hosts a Mandarin voice pack used by the A-Learn Chinese Dictionary Android app. The model runs on the device after download; voice data are not sent to a translation or speech server.

The pack repackages the existing **Kokoro-82M v1.0** model, voice vectors and Chinese frontend files distributed by **sherpa-onnx**. The model and voices were created upstream, not newly trained for A-Learn. Audio quality and Mandarin pronunciation depend on the upstream model and must be assessed on the target device.

## Downloads

Open [Releases](https://github.com/myothureinlay/a-learn-voice-models/releases) to see the voice-pack release and its asset. GitHub's release asset API reports `download_count`, a count of download requests; retries and other clients can be counted, so it is not a count of unique people or completed installs. Public release assets are accessible without a GitHub account.

The app verifies the archive size and SHA-256 before accepting an installation. The pack is ZIP_STORED to let the app write the model during download, avoiding a separate bzip2 decompression step.

## Credits and license

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0 model weights.
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), Apache-2.0 model conversion and inference project.
- Chinese lexicon generated from [Misaki](https://github.com/hexgrad/misaki) (Apache-2.0) and [pypinyin](https://github.com/mozillazg/python-pinyin) (MIT).
- Kokoro's model card attributes training data to [Koniwa tnc](https://github.com/koniwa/koniwa) (CC BY 3.0) and the [SIWIS French speech database](https://datashare.ed.ac.uk/handle/10283/2353) (CC BY 4.0); those source recordings are not included here.
- Chinese normalization data were obtained from [this upstream export](https://huggingface.co/csukuangfj/icefall-tts-aishell3-vits-low-2024-04-06); its complete input provenance has not been independently audited.

The archive includes `LICENSE`, `PYPINYIN-LICENSE`, and `NOTICE`. No eSpeak data or audio recordings are included. See upstream sources for their complete licenses and notices.
