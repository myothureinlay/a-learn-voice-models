"""Fetch the independently recorded tone references for the paired acoustic check."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json

REVISION = 'aa25ecce7b7fb02757b2c2b8e3c01aa975812edc'
URL = f'https://raw.githubusercontent.com/davinfifield/mp3-chinese-pinyin-sound/{REVISION}/mp3/'
DIRECTORY = Path(__file__).resolve().parent / 'reference'


def fetch(name):
    path = DIRECTORY / f'{name}.mp3'
    if not path.exists():
        with urlopen(URL + path.name, timeout=40) as source:
            audio = source.read()
        temporary = path.with_suffix('.partial')
        temporary.write_bytes(audio)
        temporary.replace(path)
    return path.name, hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    DIRECTORY.mkdir(exist_ok=True)
    names = [f'{syllable}{tone}' for syllable in ('ma', 'mi', 'shu', 'yi', 'ba') for tone in range(1, 5)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        digests = dict(pool.map(fetch, names))
    (DIRECTORY / 'sources.json').write_text(json.dumps({'url': URL, 'sha256': digests}, indent=2) + '\n')
