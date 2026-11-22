"""Apply the trained pitch and duration projections to a verified voicepack-v2 model."""
import argparse
import hashlib
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper


BASE_HASH = "0d73f5595f1769b499300fb099b7bd93f5c223b19bd1229511886de2c0335917"
PROJECTIONS = {
    "pitch-projection.npy": "kmodel.predictor.F0_proj.weight",
    "duration-projection.npy": "onnx::MatMul_7658",
}


def export(base, weights, output):
    if output.resolve() == base.resolve():
        raise ValueError("Choose a different output path from the v2 model")
    binary = bytearray(base.read_bytes())
    if hashlib.sha256(binary).hexdigest() != BASE_HASH:
        raise ValueError("Use the voicepack-v2 base model")
    graph = onnx.load(base).graph
    parameters = {tensor.name: tensor for tensor in graph.initializer}
    for filename, name in PROJECTIONS.items():
        tensor = parameters[name]
        old = numpy_helper.to_array(tensor)
        new = np.load(weights / filename, allow_pickle=False)
        if new.dtype != old.dtype or new.shape != old.shape or not np.isfinite(new).all():
            raise ValueError(f"Invalid projection: {filename}")
        if binary.count(tensor.raw_data) != 1:
            raise ValueError(f"Unexpected layout for {name}")
        start = binary.index(tensor.raw_data)
        binary[start:start + new.nbytes] = new.astype("<f4", copy=False).tobytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as tmp:
        path = Path(tmp.name)
    try:
        path.write_bytes(binary)
        onnx.checker.check_model(str(path))
        path.replace(output)
    finally:
        path.unlink(missing_ok=True)
    print(f"{hashlib.sha256(binary).hexdigest()}  {output.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "weights", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    export(args.base, args.weights, args.output)
