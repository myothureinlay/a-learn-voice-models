"""Condense per-run measurements from evaluate_waveforms.py and check_phrases.py."""
import argparse
import json
from pathlib import Path
from statistics import mean


def model_summary(root, name):
    isolated = json.loads((root / f"{name}-summary.json").read_text())["summary"]
    phrases = json.loads((root / f"phrase-{name}.json").read_text())
    isolated.pop("phrases")  # The dedicated aligned set measures phrase timing.
    pauses = [r for r in phrases["cases"] if r["category"] == "paused"]
    pause_errors = [r["generated_internal_quiet_ms"] - r["longest_gap_ms"] for r in pauses]
    phrase_summary = phrases["summary"]
    phrase_summary["paused"]["longest_quiet_mae_ms"] = round(mean(map(abs, pause_errors)), 1)
    return {"isolated_and_waveform": isolated, "held_out_phrase_timing": phrase_summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = {
        "protocol": {
            "voice_ids": [49, 50, 51, 52],
            "speed": 0.9,
            "silence_scale": 0.2,
            "lexicon": "v2 upstream lexicon shared by all three model runs",
            "isolated": "Five syllable families by four tones by four synthetic voices; one recording per family and tone.",
            "phrase_timing": "Reference speech span from first to last syllable. Twenty-four held-out texts with eight short, eight long and eight paused utterances. Speakers may recur across categories.",
            "interpretation": "Nearest contour and pitch-shape errors are single-reference acoustic measures. Correct pronunciation, audible clipping and naturalness require listening judgments. Clipped_samples counts full-scale PCM samples, not missing ends of words."
        },
        "models": {name: model_summary(args.directory, name) for name in ("v2", "pitch", "combined")}
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
