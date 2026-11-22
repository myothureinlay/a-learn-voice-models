"""Supervise short and sustained syllables on human durations, with per-utterance weight limits."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_timing as timing


def predict(data, delta):
    result = []
    for h, logits, starts, human in data:
        base = timing.syllables(logits, starts)
        after = timing.syllables(logits + (h @ delta)[:,None], starts)
        result.append((base, after, human))
    return result


def design(data):
    xs, ys = [], []
    for h, logits, starts, human in data:
        base = timing.syllables(logits, starts)
        p = expit(logits)
        derivative = (p*(1-p)).sum(1) * timing.FRAME / timing.SPEED
        derivative[p.sum(1)/timing.SPEED < 1] = 0
        features = np.add.reduceat(h * derivative[:,None], starts)
        # Blend relative rhythm with raw duration. Limit outliers and contribution of one reader.
        ratio = np.clip(base.sum()/human.sum(), .85, 1.15)
        target = human * (.5 + .5 * ratio)
        xs.append(features); ys.append(np.clip(target - base, -.18, .18))
    x, y = np.vstack(xs).astype("d"), np.concatenate(ys)
    gram = x.T @ x
    return np.linalg.solve(gram + .035*len(y)*np.eye(x.shape[1]), x.T @ y).astype("f")


def metrics(data, delta):
    rows = predict(data, delta)
    scaled = np.concatenate([a/a.sum()*r.sum()-r for _,a,r in rows])
    err = np.concatenate([a-r for _,a,r in rows]); durations = np.concatenate([r for _,_,r in rows])
    def mae(x):
        return round(float(np.mean(abs(x))) * 1000, 2)
    return dict(raw_ms=mae(err), relative_ms=mae(scaled),
                short_ms=mae(err[durations<=.15]), sustained_ms=mae(err[durations>=.30]),
                short_count=int(np.sum(durations<=.15)), sustained_count=int(np.sum(durations>=.30)),
                sentence_length_ratio=float(np.mean([a.sum()/b.sum() for b,a,_ in rows])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("pack", "archive", "out"):
        parser.add_argument(name, type=Path)
    args=parser.parse_args()
    pack, archive, out=args.pack, args.archive, args.out
    path=pack/'model.onnx'
    if timing.digest(path) != '0d73f5595f1769b499300fb099b7bd93f5c223b19bd1229511886de2c0335917' or timing.digest(archive)!=timing.DATA:
        raise ValueError("Unexpected weights or alignments")
    out.mkdir(exist_ok=True)
    model=onnx.load(path); params={p.name:p for p in model.graph.initializer}
    w,b=[numpy_helper.to_array(params[k]) for k in (timing.WEIGHT,timing.BIAS)]
    table, period=timing.lookup(pack)
    pools=timing.records(archive,table)
    styles=np.memmap(pack/'voices.bin',dtype='<f4',mode='r').reshape(-1,510,256)
    sess=timing.hidden_session(model)
    train,val,test=[timing.features(p,sess,styles,table,period,w,b) for p in pools]
    delta=design(train)
    scores={s:metrics(val,s*delta) for s in (0,.125,.25,.5,.75,1.)}
    def allowed(m):
        return abs(m['sentence_length_ratio']-1)<=.045 and all(m[k] <= scores[0][k]+1 for k in ('short_ms','sustained_ms','relative_ms'))
    strength=min([s for s in scores if allowed(scores[s])],key=lambda s:scores[s]['raw_ms'])
    data=(w+(strength*delta)[:,None]).astype('<f4')
    np.save(out/'duration-projection.npy',data)
    old=params[timing.WEIGHT].raw_data
    blob=path.read_bytes()
    if blob.count(old)!=1:
        raise ValueError("Unexpected weight layout")
    report=dict(val=scores, strength=strength, test_v2=metrics(test,0*delta),test_candidate=metrics(test,strength*delta),
                offset=blob.index(old), utterances=[len(p) for p in pools],train_syllables=sum(len(r[-1]) for r in pools[0]))
    (out/'duration-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    main()
