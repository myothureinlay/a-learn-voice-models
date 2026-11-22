"""Use the released acoustic graph and paired human audio to measure F0 contours."""
import hashlib,json
import sys
from pathlib import Path
import numpy as np
import onnx,onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper
import parselmouth
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_timing import lookup

H='/F0.2/Mul_output_0'
D='/Clip_output_0'
F='kmodel.predictor.F0_proj.weight'
B='kmodel.predictor.F0_proj.bias'

def extract(model_path):
    model=onnx.load(model_path)
    params={p.name:p for p in model.graph.initializer}
    weight=numpy_helper.to_array(params[F]).ravel().copy()
    bias=numpy_helper.to_array(params[B]).item()
    needed={H,D};nodes=[]
    for n in reversed(model.graph.node):
        if needed.intersection(n.output):
            nodes.append(n)
            needed.update(n.input)
            # ONNX Loop bodies can capture outer scope tensors by name.
            def capture(node):
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        for child in attr.g.node:
                            needed.update(child.input)
                            capture(child)
            capture(n)
    g=helper.make_graph(nodes[::-1],'acoustic-prefix',[v for v in model.graph.input if v.name in needed],
        [helper.make_tensor_value_info(H,TensorProto.FLOAT,[1,256,None]),helper.make_tensor_value_info(D,TensorProto.FLOAT,None)],
        [p for p in model.graph.initializer if p.name in needed])
    prefix=helper.make_model(g,opset_imports=model.opset_import);prefix.ir_version=model.ir_version
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.log_severity_level=3
    return ort.InferenceSession(prefix.SerializeToString(),opts,providers=['CPUExecutionProvider']),weight,bias

def infer(r,session,table,period,styles):
    seqs=[table[p] for _,_,p in r['syllables']]
    phones=[i for s in seqs for i in s]
    toks=np.array([[0,*phones,period,0]],dtype=np.int64)
    idx=int(hashlib.sha256(r['uid'].encode()).hexdigest()[-4:],16)%4+49
    h,d=session.run(None,{'tokens':toks,'style':styles[idx,len(phones)+1][None,:],
        'speed':np.array([.90],dtype=np.float32)})
    h=h[0].T;d=d.flatten().astype(int)
    assert len(h)==2*d.sum() and d.shape==(len(phones)+3,)
    edges=2*np.cumsum([0]+list(d))
    boundaries=np.cumsum([1]+[len(s) for s in seqs])
    return h,edges[boundaries],idx

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    args=parser.parse_args()
    session,w,b=extract(args.base/'model.onnx')
    table,period=lookup(args.base)
    styles=np.memmap(args.base/'voices.bin',dtype='<f4',mode='r').reshape(-1,510,256)
    out=args.data/'features-v2';out.mkdir(exist_ok=True)
    rows=json.loads((args.data/'manifest.json').read_text())
    n=0
    for r in rows:
        target=out/(r['uid']+'.npz')
        if target.exists():continue
        wav=args.data/'audio'/(r['uid']+'.wav')
        if not wav.exists() or any(p not in table for _,_,p in r['syllables']):continue
        h,edges,idx=infer(r,session,table,period,styles)
        hz=h@w+b
        pitch=parselmouth.Sound(str(wav)).to_pitch_ac(time_step=.01,pitch_floor=60,pitch_ceiling=360)
        t=pitch.xs();f=pitch.selected_array['frequency']
        # Map the F0 grid by syllable alignment and relative location. Keep only
        # pairs that were voiced in the recording and generated at ordinary speech pitch.
        # Store the entire predicted frame range separately for safety checks.
        data=[];hf=[];ids=[];tf=[]
        for j,(a,z,p) in enumerate(r['syllables']):
            lo,hi=edges[j:j+2]
            position=(np.arange(hi-lo)+.5)/(hi-lo)
            rt=a+position*(z-a)
            nearest=np.clip(np.rint((rt-t[0])/.01).astype(int),0,len(t)-1)
            ref=f[nearest]
            mask=(ref>=60)&(ref<=360)&(hz[lo:hi]>=55)&(hz[lo:hi]<=380)&(position>=.06)&(position<=.94)&(abs(rt-t[nearest])<=.006)
            ii=np.flatnonzero(mask)
            if len(ii)<4 or ii[-1]-ii[0]<(hi-lo)*.28:continue
            # Twelve uniformly sampled frame positions per syllable maximum, so
            # long syllables do not overwhelm short and neutral syllables.
            if len(ii)>12:ii=ii[np.rint(np.linspace(0,len(ii)-1,12)).astype(int)]
            data.extend(h[lo+ii]);hf.extend(ref[ii]);ids.extend([j]*len(ii));tf.extend(position[ii])
        if data:
            np.savez(target,x=np.array(data,'f'),ref=np.array(hf,'f'),sy=np.array(ids,'i2'),pos=np.array(tf,'f'),
                voice=np.array(idx),edges=edges,internal_f0=hz.astype('f'),silence_h=h[hz<40][::4])
            n+=1
            if n==1:print(r['uid'],h.shape,edges,np.percentile(hz,[0,10,50,90,100]),len(data),flush=True)
    print('New feature files:',n,flush=True)
if __name__=='__main__':main()
