"""Fit the final Kokoro pitch projection against pitch tracks from aligned Mandarin speech."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

F0 = "kmodel.predictor.F0_proj.weight"
BIAS = "kmodel.predictor.F0_proj.bias"


def read_split(rows, directory, split, weight, bias):
    utterances = []
    for row in rows:
        if row["split"] != split:
            continue
        path = directory / (row["uid"] + ".npz")
        if not path.exists():
            continue
        with np.load(path) as data:
            x = data["x"].astype("d")
            original = x @ weight + bias
            human = data["ref"].astype("d")
            # Match speaker register within each utterance while retaining its local tone contours.
            target = human * np.exp(np.median(np.log(original / human)))
            utterances.append((x, original, target, data["sy"], data["silence_h"].astype("d")))
    return utterances


def fit(data):
    gram = np.zeros((256, 256))
    rhs = np.zeros(256)
    frames = 0
    for x, original, target, syllables, quiet in data:
        # Log-Hz linearization keeps high-pitched syllables from dominating the fit.
        design = x / original[:, None]
        residual = np.clip(np.log(target / original), -.45, .45)
        gram += design.T @ design
        rhs += design.T @ residual
        adjacent = syllables[1:] == syllables[:-1]
        slope = np.diff(design, axis=0)[adjacent]
        gram += 2 * slope.T @ slope
        rhs += 2 * slope.T @ np.diff(residual)[adjacent]
        gram += (quiet.T @ quiet) / (65 ** 2)
        frames += len(target)
    ridge = .02 * np.trace(gram) / len(rhs)
    return np.linalg.solve(gram + ridge * np.eye(len(rhs)), rhs), frames


def score(data, correction):
    contour, slope, register = [], [], []
    for x, original, target, syllables, _ in data:
        hz = original + x @ correction
        if np.min(hz) <= 0:
            return dict(contour_st=1e6, slope_st=1e6, register_shift_st=1e6)
        error = 12 * np.log2(hz / target)
        contour.extend(abs(error))
        slope.extend(abs(np.diff(error)[syllables[1:] == syllables[:-1]]))
        register.append(abs(np.median(12 * np.log2(hz / original))))
    return dict(contour_st=float(np.mean(contour)), slope_st=float(np.mean(slope)),
                register_shift_st=float(np.mean(register)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "manifest", "features", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    with args.model.open("rb") as stream:
        base_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if base_hash != "0d73f5595f1769b499300fb099b7bd93f5c223b19bd1229511886de2c0335917":
        raise ValueError("Use the voicepack-v2 base model")
    model = onnx.load(args.model)
    params = {p.name: p for p in model.graph.initializer}
    array = numpy_helper.to_array(params[F0])
    weight = array.flatten().astype("d")
    bias = numpy_helper.to_array(params[BIAS]).item()
    rows = json.loads(args.manifest.read_text())
    train, val, test = [read_split(rows, args.features, s, weight, bias) for s in ("train","val","test")]
    if [len(group) for group in (train, val, test)] != [201, 27, 33]:
        raise ValueError("Incomplete feature coverage for the pinned corpus")
    delta, frames = fit(train)
    scores = {s: score(val, s*delta) for s in (0., .125, .25, .5, .75, 1.)}
    eligible = [s for s, v in scores.items() if v["register_shift_st"] <= .5 and v["slope_st"] <= scores[0]["slope_st"] + .02]
    strength = min(eligible, key=lambda s: scores[s]["contour_st"])
    updated = (weight + strength * delta).astype("<f4").reshape(array.shape)
    args.out.mkdir(exist_ok=True, parents=True)
    np.save(args.out/"pitch-projection.npy", updated)
    original_bytes = args.model.read_bytes()
    old = params[F0].raw_data
    if original_bytes.count(old) != 1 or len(old) != updated.nbytes:
        raise ValueError("Unexpected projection in ONNX file")
    candidate = original_bytes.replace(old, updated.tobytes())
    (args.out/"model.onnx").write_bytes(candidate)
    report = {"training_frames":frames, "utterances":[len(x) for x in (train,val,test)],
              "validation":scores, "strength":strength, "test_v2":score(test,0*delta),
              "test_candidate":score(test,strength*delta), "offset":original_bytes.index(old),
              "source_sha256":hashlib.sha256(original_bytes).hexdigest(),
              "sha256":hashlib.sha256(candidate).hexdigest()}
    (args.out/"pitch-results.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
