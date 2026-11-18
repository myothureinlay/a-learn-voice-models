"""Fit a low-rank update to the Mandarin Kokoro duration projection on AISHELL-3 alignments."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import tarfile
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
from pypinyin import lazy_pinyin, Style
from scipy.special import expit

BASE = "b40f62b166ac8164b0627ef48a0b358eda0985e272fb03ef5252e7206305da11"
DATA = "62a9bfe32eceac17719fab175469c5c2ca2486dfa20124d814ab2b144e098789"
HIDDEN = "/lstm/Transpose_2_output_0"
WEIGHT = "onnx::MatMul_7658"
BIAS = "kmodel.predictor.duration_proj.linear_layer.bias"
SPEED = .9
FRAME = .025
INTERVAL = re.compile(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"')
MARKS = "→↗↓↘"
SIZES = (800, 180, 300)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lookup(base):
    ids = {p:int(i) for p,i in (line.rsplit(' ',1) for line in (base/'tokens.txt').read_text().splitlines())}
    counts = defaultdict(Counter)
    for line in (base/'lexicon-zh.txt').read_text().splitlines():
        char, *phones = line.split()
        if len(char) != 1 or not phones or not all(x in ids for x in phones):
            continue
        py = lazy_pinyin(char, style=Style.TONE3, neutral_tone_with_five=True)[0].replace('ü','v')
        if not re.fullmatch('[a-zv]+[1-5]', py):
            continue
        marked = [p for p in phones if p in MARKS]
        if marked != list(MARKS[int(py[-1])-1:int(py[-1])]):
            continue
        counts[py][tuple(ids[p] for p in phones)] += 1
    # Neutral syllables use the same segmental pronunciation without a tone token.
    table = {py:hist.most_common(1)[0][0] for py,hist in counts.items()}
    for py,seq in list(table.items()):
        table.setdefault(py[:-1]+'5', tuple(x for x in seq if x not in {ids[m] for m in MARKS}))
    return table, ids['.']


def records(archive, table):
    pools = [[], [], []]
    with tarfile.open(archive, 'r|gz') as src:
        for member in src:
            if not member.name.endswith('.TextGrid'):
                continue
            path = Path(member.name)
            spk = path.parent.name
            speaker_bucket = int(hashlib.sha256(spk.encode()).hexdigest()[:8],16) % 10
            split = 0 if speaker_bucket < 7 else 1 if speaker_bucket==7 else 2
            # Hash-order sampling makes the speaker and utterance splits reproducible.
            key = hashlib.sha256(path.stem.encode()).hexdigest()
            if key[:2] > '4f':
                continue
            raw = src.extractfile(member).read().decode('utf-8').split('name = "phones"')[0]
            parts = [(float(b)-float(a), p.replace('ü','v')) for a,b,p in INTERVAL.findall(raw)]
            while parts and not parts[0][1]:
                parts.pop(0)
            while parts and not parts[-1][1]:
                parts.pop()
            if not 5 <= len(parts) <= 24 or any(p not in table or not .045 <= d <= .70 for d,p in parts):
                continue
            pools[split].append((key, spk, path.stem, parts))
    used = set()
    result = []
    for pool, size in zip(pools,SIZES):
        subset = []
        for key, spk, uid, parts in sorted(pool):
            text = tuple(p for _,p in parts)
            if text in used:
                continue
            subset.append((key,spk,uid,parts))
            used.add(text)
            if len(subset)==size:
                break
        if len(subset)!=size:
            raise ValueError("Insufficient eligible, unique utterances")
        result.append(subset)
    return result


def hidden_session(model):
    # Extract the prefix without executing the vocoder or requiring autograd through ONNX.
    needed = {HIDDEN}
    nodes = []
    for node in reversed(model.graph.node):
        if needed.intersection(node.output):
            nodes.append(node)
            needed.update(node.input)
    graph = helper.make_graph(list(reversed(nodes)), 'duration-features',
        [v for v in model.graph.input if v.name in needed],
        [helper.make_tensor_value_info(HIDDEN,TensorProto.FLOAT,[1,None,None])],
        [i for i in model.graph.initializer if i.name in needed])
    prefix = helper.make_model(graph, opset_imports=model.opset_import)
    prefix.ir_version = model.ir_version
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(prefix.SerializeToString(),opts,providers=['CPUExecutionProvider'])


def features(rows, session, styles, table, period, W, b):
    results = []
    for key,spk,uid,parts in rows:
        seqs = [table[p] for _,p in parts]
        seq = [i for s in seqs for i in s]
        toks = np.array([[0, *seq, period, 0]],dtype=np.int64)
        sid = 49 + int(key[-4:],16)%4
        h = session.run(None,{'tokens':toks,'style':styles[sid,len(seq)+1][None,:]})[0][0,1:1+len(seq)]
        boundaries = np.cumsum([0]+[len(s) for s in seqs[:-1]])
        logits = h@W + b
        target = np.array([d for d,_ in parts])
        results.append((h,logits,boundaries,target))
    return results


def syllables(logits,starts):
    frames = np.maximum(1,np.rint(expit(logits).sum(axis=1)/SPEED))
    return np.add.reduceat(frames,starts)*FRAME


def fit(train):
    xs,ys=[],[]
    for h,logits,start,seconds in train:
        base = syllables(logits,start)
        target = seconds / seconds.sum() * base.sum()
        probs = expit(logits)
        derivative = (probs*(1-probs)).sum(1)/SPEED*FRAME
        derivative[probs.sum(1)/SPEED<1] = 0
        xs.append(np.add.reduceat(h*derivative[:,None],start,axis=0))
        ys.append(target-base)
    x,y=np.vstack(xs).astype('d'),np.concatenate(ys)
    gram=x.T@x
    ridge=0.025*len(x)
    return np.linalg.solve(gram+ridge*np.eye(x.shape[1]),x.T@y).astype('f')


def metrics(rows,v):
    relative, absolute, durations = [], [], []
    for h,logits,start,seconds in rows:
        estimate = syllables(logits+(h@v)[:,None],start)
        baseline = syllables(logits,start)
        relative.extend(abs(estimate/estimate.sum()*seconds.sum()-seconds))
        absolute.extend(abs(estimate-seconds))
        durations.append(estimate.sum()/baseline.sum())
    return dict(syllables=len(relative), normalized_mae_ms=round(float(np.mean(relative))*1000,3),
        raw_mae_ms=round(float(np.mean(absolute))*1000,3),
        mean_duration_ratio=round(float(np.mean(durations)),4))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path); parser.add_argument('alignments',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    base,archive,out=args.base,args.alignments,args.output
    out.mkdir(parents=True,exist_ok=True)
    if digest(base/'model.onnx') != BASE or digest(archive) != DATA:
        raise ValueError('Use the pinned model and alignment archive')
    model=onnx.load(base/'model.onnx')
    params={x.name:x for x in model.graph.initializer}
    W,b=[numpy_helper.to_array(params[k]) for k in (WEIGHT,BIAS)]
    session=hidden_session(model)
    table,period=lookup(base)
    pools=records(archive,table)
    styles=np.memmap(base/'voices.bin',dtype='<f4',mode='r').reshape(-1,510,256)
    data=[features(p,session,styles,table,period,W,b) for p in pools]
    v=fit(data[0]); zero=np.zeros_like(v)
    # Choose update strength using validation speakers only. Keep the original as a candidate.
    scales=(0,.25,.5,.75,1.)
    scores={s:metrics(data[1],v*s) for s in scales}
    strength=min(scales,key=lambda s:scores[s]['normalized_mae_ms'])
    baseline,adapted=[metrics(data[2],u) for u in (zero,v*strength)]
    report={'split_utterances':list(SIZES),'split_speakers':[len({r[1] for r in p}) for p in pools],
        'split_syllables':[sum(len(r[3]) for r in p) for p in pools],'voice_ids':[49,50,51,52],
        'base_model_sha256':BASE,'alignment_sha256':DATA,'validation':scores,'strength':strength,
        'held_out_before':baseline,'held_out_after':adapted}
    # Preserve the original byte layout so an installed pack can be updated transactionally.
    original=params[WEIGHT].raw_data
    updated=np.asarray(W+(strength*v)[:,None],dtype='<f4').tobytes()
    binary=(base/'model.onnx').read_bytes()
    if binary.count(original)!=1 or len(updated)!=len(original):
        raise ValueError('Unexpected ONNX weight layout')
    at=binary.index(original)
    (out/'model.onnx').write_bytes(binary[:at]+updated+binary[at+len(original):])
    report.update(weight_offset=at,weight_bytes=len(updated),model_sha256=digest(out/'model.onnx'))
    (out/'duration-projection.bin').write_bytes(updated)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
