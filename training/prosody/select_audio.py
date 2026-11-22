"""Select a speaker- and transcript-disjoint sample from pinned AISHELL-3 human recordings."""
from pathlib import Path
import hashlib,json,re,tarfile
from collections import defaultdict
import concurrent.futures, subprocess
REV='f20d5db4a31fe779ef07bb1af4ea92da5c786622'
SHA='62a9bfe32eceac17719fab175469c5c2ca2486dfa20124d814ab2b144e098789'
HUB='https://huggingface.co/datasets/AISHELL/AISHELL-3'
ALIGN='https://paddlespeech.cdn.bcebos.com/MFA/AISHELL-3/with_tone/aishell3_alignment_tone.tar.gz'
I=re.compile(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"')
def digest(x):
    return hashlib.sha256(x.encode()).hexdigest()
def prepare():
    sources = [
        ('align.tar.gz', ALIGN, SHA),
        ('trans.txt', f'{HUB}/resolve/{REV}/train/content.txt', 'ed70f67d5558124ee22e609260177eaf20aa63391dd9df8df0a7e22b9213aca0'),
        ('spk-info.txt', f'{HUB}/resolve/{REV}/spk-info.txt', '86148f95beab318ad9d08bf945b1d56ffba8c97b09137ae784cc62b0cd6539ef'),
        ('revision.json', f'https://huggingface.co/api/datasets/AISHELL/AISHELL-3/revision/{REV}', None),
    ]
    for name, url, sha in sources:
        path = Path(name)
        if not path.is_file():
            temp = path.with_suffix(path.suffix + '.tmp')
            subprocess.run(['curl','--retry','3','-fsSL',url,'-o',str(temp)],check=True)
            temp.replace(path)
        if sha:
            with path.open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest() != sha:
                    raise ValueError(f'Source hash mismatch: {name}')
        elif json.loads(path.read_text())['sha'] != REV:
            raise ValueError('AISHELL-3 revision mismatch')
def select():
    with open('align.tar.gz','rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==SHA
    speakers={p[0] for s in open('spk-info.txt') if len(p:=s.split())==4 and p[2]=='male'}
    trans={p[0].split('.')[0]:p[1].split() for s in open('trans.txt') if len(p:=s.strip().split('\t'))==2}
    available={Path(item['rfilename']).stem for item in json.loads(Path('revision.json').read_text())['siblings'] if item['rfilename'].startswith('train/wav/')}
    by=defaultdict(list)
    with tarfile.open('align.tar.gz','r|gz') as tar:
        for m in tar:
            uid=Path(m.name).stem; speaker=uid[:7]
            if speaker not in speakers or uid not in available or uid not in trans or not m.name.endswith('.TextGrid'):continue
            raw=tar.extractfile(m).read().decode().split('name = "phones"')[0]
            words=[(round(float(a),4),round(float(b),4),p.replace('ü','v')) for a,b,p in I.findall(raw) if p]
            if not 6<=len(words)<=25:continue
            chars,py=trans[uid][::2],trans[uid][1::2]
            if len(chars)!=len(words) or any(len(h)!=1 for h in chars):continue
            if any(p!=alternative.replace('ü','v') or not .045<=b-a<=.75 for (a,b,p),alternative in zip(words,py)):continue
            # A long internal pause is real speech, not a sustained syllable. Do not use
            # those boundaries to supervise the acoustic head when no punctuation is supplied.
            if any(words[k+1][0]-words[k][1]>.13 for k in range(len(words)-1)):continue
            by[speaker].append(dict(uid=uid,speaker=speaker,text=''.join(chars),syllables=words))
    used=set();rows=[]
    for sp,rr in sorted(by.items(),key=lambda x:digest(x[0])):
        b=int(digest(sp)[:8],16)%10;s='train' if b<7 else 'val' if b==7 else 'test'
        selected=0
        for r in sorted(rr,key=lambda r:digest('pitch-v3:'+r['uid'])):
            key=tuple(x[2] for x in r['syllables'])
            if key in used:continue
            used.add(key);r['split']=s;rows.append(r);selected+=1
            if selected>=dict(train=14,val=14,test=18)[s]:break
    return rows
def fetch(row):
    uid,sp=row['uid'],row['speaker'];p=Path('audio')/(uid+'.wav')
    if p.exists() and p.stat().st_size>2000:return
    url=f'https://huggingface.co/datasets/AISHELL/AISHELL-3/resolve/{REV}/train/wav/{sp}/{uid}.wav'
    subprocess.run(['curl','--retry','3','--connect-timeout','12','--max-time','45','-fsSL',url,'-o',str(p)+'.tmp'],check=True)
    tmp=p.with_suffix('.wav.tmp')
    with tmp.open('rb') as f:
        if f.read(4)!=b'RIFF':raise ValueError('unexpected audio')
    tmp.rename(p)
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'select', 'fetch'))
    action = parser.parse_args().action
    if action=='prepare':
        prepare()
    if action=='select':
        rows=select();Path('manifest.json').write_text(json.dumps(rows,ensure_ascii=False))
        for s in ('train','val','test'):
            selected=[r for r in rows if r['split']==s];print(s,len(selected),len(set(r['speaker'] for r in selected)),sum(len(r['syllables']) for r in selected))
    if action=='fetch':
        Path('audio').mkdir(exist_ok=True);rows=json.loads(Path('manifest.json').read_text())
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            for _ in pool.map(fetch,rows):pass
