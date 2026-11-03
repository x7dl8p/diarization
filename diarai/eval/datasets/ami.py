"""Build the AMI test-set manifest: Mix-Headset audio + speaker RTTMs + per-speaker word transcripts.

Reference RTTMs come from pyannote/AMI-diarization-setup ("only_words" variant, the standard
AMI diarization benchmark); transcripts come from the AMI manual annotations (words layer).

Usage:
    python -m diarai.eval.datasets.ami --raw data/raw --out data/ami
    python -m diarai.eval.datasets.ami --raw data/raw --out data/ami_dev --split dev   # meetings that were downloaded
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import soundfile as sf

NITE = "{http://nite.sourceforge.net/}"


def speaker_map(meetings_xml: Path):
    """{meeting: {nxt_agent_letter: global_name}}"""
    out = {}
    for m in ET.parse(meetings_xml).getroot().iter("meeting"):
        out[m.get("observation")] = {s.get("nxt_agent"): s.get("global_name") for s in m.iter("speaker")}
    return out


def read_words(words_xml: Path):
    words = []
    for w in ET.parse(words_xml).getroot().iter("w"):
        if w.get("punc") == "true" or w.get("starttime") is None or not (w.text or "").strip():
            continue
        words.append((float(w.get("starttime")), float(w.get("endtime")), w.text.strip()))
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/ami"))
    ap.add_argument("--max-gap", type=float, default=1.0, help="split a speaker's words into segments at gaps > this")
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    args = ap.parse_args()

    annot = args.raw / "ami_annot" / "x"
    spk = speaker_map(annot / "corpusResources" / "meetings.xml")
    (args.out / "ref").mkdir(parents=True, exist_ok=True)
    manifest = []
    suffix = "" if args.split == "test" else "_dev"
    for meeting in (args.raw / "ami_setup" / f"{args.split}.meetings.txt").read_text().split():
        audio = args.raw / f"ami{suffix}_audio" / f"{meeting}.Mix-Headset.wav"
        if not audio.exists():
            print(f"skip {meeting}: no audio")
            continue
        segments = []
        for letter, name in spk[meeting].items():
            path = annot / "words" / f"{meeting}.{letter}.words.xml"
            if not path.exists():
                continue
            cur = None
            for s, e, t in read_words(path):
                if cur and s - cur["end"] <= args.max_gap:
                    cur["end"], cur["text"] = max(cur["end"], e), cur["text"] + " " + t
                else:
                    cur = {"speaker": name, "start": s, "end": e, "text": t}
                    segments.append(cur)
        segments.sort(key=lambda x: x["start"])
        ref = {"uri": meeting, "speakers": sorted(spk[meeting].values()), "segments": segments}
        (args.out / "ref" / f"{meeting}.json").write_text(json.dumps(ref, indent=1))
        rttm = args.out / "ref" / f"{meeting}.rttm"
        rttm.write_text((args.raw / f"ami{suffix}_setup" / f"{meeting}.rttm").read_text())
        info = sf.info(audio)
        manifest.append({"uri": meeting, "audio": str(audio), "rttm": str(rttm),
                         "transcript": str(args.out / "ref" / f"{meeting}.json"),
                         "num_speakers": len(spk[meeting]), "duration": round(info.duration, 2)})
        print(f"{meeting}: {len(segments)} segments, {info.duration / 60:.1f} min")
    (args.out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))


if __name__ == "__main__":
    main()
