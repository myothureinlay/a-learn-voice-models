"""Paired end-to-end Mandarin waveform checks. Values are acoustic proxies, not listening scores."""
import argparse
from pathlib import Path
import json
import subprocess
import shutil

import numpy as np
import parselmouth
import sherpa_onnx
from scipy.io.wavfile import write
from scipy.signal import butter, sosfiltfilt

FAMILIES = {'ma': '妈麻马骂', 'mi': '咪迷米密', 'shu': '书赎鼠树', 'yi': '依姨椅义', 'ba': '八拔把坝'}
PHRASES = {'short': '走。', 'neutral': '妈妈来了。', 'sandhi': '你好，我想买两把雨伞。',
    'nobreak': '我们今天去公园散步。', 'comma': '我们今天，去公园散步。',
    'long': '我们一起去银行取钱，然后回家看看妈妈。',
    'question': '你今天要不要一起去公园？', 'released': '春天到了，我们一起去公园散步吧。'}
POSITIONS = np.linspace(.10, .90, 9)


def measure(sound):
    y = sound.values.mean(axis=0)
    rate = sound.sampling_frequency
    # The MP3 references carry low-frequency recorder bias. Remove it before
    # deciding where speech starts and ends, keeping the unmodified PCM for peak checks.
    filtered = sosfiltfilt(butter(2, 65, fs=rate, btype='highpass', output='sos'), y)
    hop = round(.005 * rate); win = round(.015 * rate)
    rms = np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(filtered, win)[::hop]**2, axis=1))
    peak = np.percentile(rms, 95)
    noise_threshold = min(peak*.25, np.percentile(rms,10)*3)
    active = rms >= max(peak * 10**(-35/20), noise_threshold, 1e-5)
    idx = np.flatnonzero(active)
    if not len(idx): raise ValueError('silent speech')
    freq = parselmouth.Sound(filtered, rate).to_pitch_ac(time_step=.005, pitch_floor=65, pitch_ceiling=400)
    f0 = freq.selected_array['frequency']; voiced = f0 > 0
    # Avoid extrapolation across disconnected voicing islands or unsupported edges.
    energy_bin = np.clip(np.rint((freq.xs()*rate-win/2)/hop).astype(int),0,len(active)-1)
    voiced &= active[energy_bin]
    pitch_t = freq.xs()[voiced]; hz = f0[voiced]; contour = []
    if len(hz) >= 5:
        t = (pitch_t - pitch_t[0]) / (pitch_t[-1] - pitch_t[0])
        for point in POSITIONS:
            mask = np.abs(t - point) <= .07
            value = np.median(hz[mask]) if np.any(mask) else np.interp(point, t, hz)
            contour.append(round(float(12*np.log2(value)),4))
    pauses = np.diff(idx) - 1
    return {'file_ms':round(len(y)/rate*1000), 'active_span_ms':round((idx[-1]*hop+win-idx[0]*hop)/rate*1000),
        'internal_quiet_ms':int(pauses.max(initial=0)*5), 'leading_quiet_ms':round(idx[0]*hop/rate*1000),
        'tail_quiet_ms':round(max(0, len(y)-idx[-1]*hop-win)/rate*1000),
        'max_absolute_amplitude':round(float(np.max(np.abs(y))),5), 'clipped_samples':int(np.sum(np.abs(y)>=32767/32768)),
        'voiced_ms':round(len(hz)*5), 'contour_st':contour}


def get_engine(base, model, silence_scale=.2):
    settings = sherpa_onnx.OfflineTtsKokoroModelConfig(model=str(model), voices=str(base/'voices.bin'),
        tokens=str(base/'tokens.txt'), lexicon=str(base/'lexicon-zh.txt'), lang='zh',
        data_dir=str(base/'espeak-ng-data') if (base/'espeak-ng-data').is_dir() else '')
    config=sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(kokoro=settings,num_threads=1),
        rule_fsts=','.join(str(base/f'{k}-zh.fst') for k in ('date','number','phone')), silence_scale=silence_scale)
    return sherpa_onnx.OfflineTts(config)


def produce(args):
    args.out.mkdir(parents=True,exist_ok=True)
    case_list = [(f'{family}{tone}', char+'。') for family, syllables in FAMILIES.items() for tone,char in enumerate(syllables,1)]
    results = {}
    engine = get_engine(args.base, args.model, args.silence_scale)
    for sid in args.voices:
        by_voice = {}
        for label, text in case_list+list(PHRASES.items()):
            audio = engine.generate(text,sid=sid,speed=.9)
            wave = np.asarray(audio.samples,dtype=np.float32)
            wav_path = args.out / f'{sid}_{label}.wav'
            write(wav_path, int(audio.sample_rate), wave)
            by_voice[label] = measure(parselmouth.Sound(wave, audio.sample_rate))
        results[str(sid)] = by_voice
    (args.out/'metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')


def reference(args):
    args.out.mkdir(parents=True,exist_ok=True)
    refs = {}
    for family in FAMILIES:
        for tone in range(1,5):
            key = f'{family}{tone}'; source=args.mp3/(key+'.mp3')
            if not source.is_file(): raise ValueError(str(source))
            target=args.out/(key+'.wav')
            if not target.exists():
                if shutil.which('afconvert'):
                    command = ['afconvert', '-f', 'WAVE', '-d', 'LEI16', str(source), str(target)]
                else:
                    command = ['ffmpeg', '-loglevel', 'error', '-i', str(source), '-c:a', 'pcm_s16le', str(target)]
                subprocess.run(command, check=True)
            refs[key] = measure(parselmouth.Sound(str(target)))
    (args.out/'metrics.json').write_text(json.dumps(refs,indent=2)+'\n')


def remeasure(args):
    by_voice = {}
    for path in sorted(args.audio.glob('*_*.wav')):
        voice, key = path.stem.split('_',1)
        by_voice.setdefault(voice,{})[key]=measure(parselmouth.Sound(str(path)))
    if not by_voice or any(len(v)!=sum(len(s) for s in FAMILIES.values())+len(PHRASES) for v in by_voice.values()):
        raise ValueError('Incomplete rendered test set')
    (args.audio/'metrics.json').write_text(json.dumps(by_voice,indent=2)+'\n')


def summarize(args):
    ref=json.loads(args.ref.read_text()); gen=json.loads(args.gen.read_text())
    shape=[]; joint=[]; span=[]; decisions=[]; rows=[]; phrase=[]
    for sid, audio in gen.items():
        for family in FAMILIES:
            ids=[f'{family}{n}' for n in range(1,5)]
            real=np.array([ref[k]['contour_st'] for k in ids]); synth=np.array([audio[k]['contour_st'] for k in ids]);
            if real.shape!=(4,9) or synth.shape!=(4,9): raise ValueError(f'Incomplete F0 coverage at {sid}/{family}')
            real-=np.median(real); synth-=np.median(synth)
            for tone,k in enumerate(ids):
                dif=synth[tone]-real[tone]; shaped=np.mean(abs(dif-np.median(dif))); total=np.mean(abs(dif))
                span_error=audio[k]['active_span_ms']-ref[k]['active_span_ms']
                hit=np.argmin(abs(synth[tone][None,:]-real).mean(axis=1))==tone
                rows.append(dict(sid=sid,key=k,shape_mae_st=float(shaped),joint_mae_st=float(total),duration_error_ms=span_error,closest_tone=bool(hit)))
                shape.append(shaped); joint.append(total); span.append(span_error); decisions.append(hit)
        phrase.extend([audio[k] for k in PHRASES]);
    summary={'syllables':len(rows),'phrase_outputs':len(phrase),'centered_f0_shape_mae_st':float(np.mean(shape)),
        'family_centered_f0_mae_st':float(np.mean(joint)),'nearest_reference_tone_count':int(np.sum(decisions)),
        'active_duration_mae_ms':float(np.mean(np.abs(span))),'mean_duration_bias_ms':float(np.mean(span)),
        'phrases': {sid:{k:audio[k]['active_span_ms'] for k in PHRASES} for sid,audio in gen.items()},
        'all_outputs_clipped_samples':sum(a['clipped_samples'] for voice in gen.values() for a in voice.values()),
        'per_tone_centered_shape_mae_st':{str(n):round(float(np.mean([row['shape_mae_st'] for row in rows if row['key'].endswith(str(n))])),4) for n in range(1,5)}}
    args.out.write_text(json.dumps({'summary':summary,'rows':rows},indent=2)+'\n'); print(json.dumps({k:v for k,v in summary.items() if k!='phrases'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); modes=parser.add_subparsers(dest='mode',required=True)
    p=modes.add_parser('synth'); p.add_argument('base',type=Path); p.add_argument('model',type=Path); p.add_argument('out',type=Path)
    p.add_argument('--voices',type=int,nargs='+',default=[49,50,51,52])
    p.add_argument('--silence-scale',type=float,default=.2)
    p=modes.add_parser('reference'); p.add_argument('mp3',type=Path); p.add_argument('out',type=Path)
    p=modes.add_parser('remeasure'); p.add_argument('audio',type=Path)
    p=modes.add_parser('compare'); p.add_argument('ref',type=Path); p.add_argument('gen',type=Path); p.add_argument('out',type=Path)
    a=parser.parse_args(); {'synth':produce,'reference':reference,'remeasure':remeasure,'compare':summarize}[a.mode](a)
