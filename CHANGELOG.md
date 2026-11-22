# Version history · 版本记录

## 2026-11-23 · Android 2.0.2 (5) · Mandarin voice preview · 普通话语音试听版

- Fixed selected dictionary readings and added third-tone sandhi for 639 more two-character words. Both voice modes use the corrections.
- 修正所选词条的读音，另为 639 个双字词补充三声变调。两种语音模式均使用修正后的读音。
- Added a reversible offline comparison of the released model and a new pitch-and-duration fine-tune. An independent four-voice test found better average contours for tones 1–3, worse fourth-tone contours, and shorter isolated syllables. Original is the default.
- 新增离线原版与微调版切换。基频与时长微调后，四种音色的一至三声音高走势平均改善，四声偏差增大，单音节偏短。默认使用原版。

[Android APK and listening clips / 安卓安装包与试听](https://github.com/myothureinlay/a-learn-voice-models/releases/tag/mandarin-voice-preview-2.0.2) · [Results / 测试结果](training/prosody/README.md)

## 2026-11-19 · Mandarin Rhythm 2 · Android 2.0.1

- Fine-tuned the syllable-duration layer for the four offline Mandarin voices. On 300 held-out utterances, timing error at matched speaking rate fell from 92.8 to 77.1 ms. Timbre and pitch weights are inherited from Kokoro. Listening tests by Mandarin speakers are pending.
- 微调四种离线普通话音色的音节时长层。在 300 条独立测试句上，按参考语速归一化后的时长误差由 92.8 降至 77.1 毫秒。音色和基频权重沿用 Kokoro，普通话听众的自然度评价尚待进行。
- Fixed isolated dictionary entries using the default character reading even when the entry supplied another pronunciation. The update handles 乐／樂 (yuè), 行 (háng), 长／長 (cháng), 重 (chóng) and 还／還 (huán). Sentence playback still uses context.
- 修正单字词条朗读忽略指定拼音的问题，覆盖乐／樂（yuè）、行（háng）、长／長（cháng）、重（chóng）及还／還（huán）。整句仍按上下文朗读。
- Corrected the old Study Update 8 label and outdated voice setup instructions. Added bilingual history in Settings → About. Existing v1 users can apply the embedded update offline under Settings → Mandarin Voices → Update voice rhythm. Allow 350 MB of temporary free space. Voice selections carry over.
- 修正设置中的旧版本标签及过时的语音说明，在设置的“关于”中加入双语版本记录。已安装 v1 的用户可在普通话语音设置中点选“Update voice rhythm”离线升级，需预留 350 MB 临时空间，原音色选择保留。

## 2026-11-14 · Voice pack 1 · 语音包 1

- Published the original four Mandarin presets with an on-device installer and GitHub download counter.
- 发布首版四种普通话预设音色，支持设备端安装并通过 GitHub 查看下载请求数。
