"""Measure phrase spans and internal quiet against independent AISHELL-3 alignments."""
import argparse
import hashlib
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import parselmouth

from evaluate_waveforms import get_engine, measure


INTERVAL = re.compile(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"')


def select(data):
    transcripts = {p[0].split('.')[0]: p[1].split() for line in (data/'trans.txt').read_text().splitlines()
                   if len(p := line.strip().split('\t')) == 2}
    manifest = json.loads((data/'manifest.json').read_text())
    exclude = {r['text'] for r in manifest}
    exclude_readings = {tuple(s[2] for s in r['syllables']) for r in manifest}
    choices = defaultdict(list)
    with tarfile.open(data/'align.tar.gz', 'r|gz') as archive:
        for item in archive:
            uid = Path(item.name).stem
            spk = uid[:7]
            if uid not in transcripts or int(hashlib.sha256(spk.encode()).hexdigest()[:8],16)%10<8:
                continue
            text = transcripts[uid][::2]
            reading = transcripts[uid][1::2]
            if not all(len(ch)==1 and '\u3400'<=ch<='\u9fff' for ch in text):
                continue
            if ''.join(text) in exclude or tuple(p.replace('ü','v') for p in reading) in exclude_readings:
                continue
            tier = archive.extractfile(item).read().decode().split('name = "phones"')[0]
            words = [(float(a), float(b), p) for a,b,p in INTERVAL.findall(tier) if p]
            if [w[2] for w in words] != reading or len(words)<2:
                continue
            gaps = [w2[0]-w1[1] for w1,w2 in zip(words,words[1:])]
            gap = max(gaps)
            if .2<=gap<=.8 and 6<=len(words)<=22:
                category = 'paused'
            elif gap<.10 and 2<=len(words)<=5:
                category = 'short'
            elif gap<.10 and 18<=len(words)<=25:
                category = 'long'
            else:
                continue
            # Supply commas at annotated breaks, since the source transcript has no punctuation.
            sentence = ''.join(ch+('，' if i<len(gaps) and gaps[i]>=.2 else '') for i,ch in enumerate(text))+'。'
            choices[category].append(dict(uid=uid, speaker=spk, text=sentence, span_ms=round(1000*(words[-1][1]-words[0][0])),
                                          longest_gap_ms=round(1000*gap), category=category))
    out=[]
    for category in ('short','long','paused'):
        speakers=set()
        for r in sorted(choices[category],key=lambda r:hashlib.sha256(('independent-timing-'+r['uid']).encode()).digest()):
            if r['speaker'] in speakers:
                continue
            speakers.add(r['speaker'])
            out.append(r)
            if len(speakers)==8:
                break
        if len(speakers)<8:
            raise RuntimeError(f'Insufficient held-out speakers for {category}')
    return out


def measure_cases(args):
    cases = json.loads(args.cases.read_text())
    engine = get_engine(args.base, args.model, args.silence_scale)
    for idx,row in enumerate(cases):
        sid=49+idx%4
        audio=engine.generate(row['text'],sid,.9)
        result=measure(parselmouth.Sound(np.asarray(audio.samples), audio.sample_rate))
        row.update(voice=sid,generated_span_ms=result['active_span_ms'],generated_internal_quiet_ms=result['internal_quiet_ms'])
    summary={}
    for cat in ('short','long','paused'):
        rows=[r for r in cases if r['category']==cat]
        errors=[r['generated_span_ms']-r['span_ms'] for r in rows]
        summary[cat]=dict(n=len(rows),span_mae_ms=round(np.mean(np.abs(errors)),1),span_bias_ms=round(np.mean(errors),1))
        if cat=='paused':
            summary[cat]['median_reference_pause_ms']=np.median([r['longest_gap_ms'] for r in rows])
            summary[cat]['median_generated_longest_quiet_ms']=np.median([r['generated_internal_quiet_ms'] for r in rows])
    return dict(summary=summary, cases=cases)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    p=parser.add_subparsers(dest='mode',required=True)
    s=p.add_parser('select');s.add_argument('data',type=Path)
    m=p.add_parser('measure');m.add_argument('base', type=Path);m.add_argument('model',type=Path);m.add_argument('cases',type=Path)
    m.add_argument('--silence-scale',type=float,default=.2)
    for sub in [s,m]:
        sub.add_argument('out',type=Path)
    args=parser.parse_args()
    result = select(args.data) if args.mode=='select' else measure_cases(args)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
