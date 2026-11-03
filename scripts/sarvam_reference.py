"""Get Sarvam AI (saaras:v4) transcripts + diarization for the real-audio clips, to use as a reference.

Runs in its own venv (the Sarvam SDK's deps clash with DiariZen's pins):
    set -a; source .env; set +a
    .sarvam-venv/bin/python scripts/sarvam_reference.py

Writes data/real/ref/<uri>.sarvam.json (raw API output) and data/real/manifest.jsonl.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from sarvamai import SarvamAI

ROOT = Path(__file__).resolve().parent.parent
CLIPS = {  # uri: (file in realaudio/, Sarvam language code)
    "podcast_hi": ("6x30ml-or-3x60ml-phd-subject-fakepodcast-128-ytshorts.savetube.me.mp3", "hi-IN"),
    "comedy_hi": ("usa-vs-india-innovation-flying-cars-comedy-128-ytshorts.savetube.me.mp3", "hi-IN"),
    "agent_en": ("how-to-catch-the-agent-mistakes-you-keep-missing-128-ytshorts.savetube.me.mp3", "en-IN"),
    "meta_en": ("what-metas-staff-engineers-do-differently-128-ytshorts.savetube.me.mp3", "en-IN"),
}


def run_job(client, files, language, outdir):
    job = client.speech_to_text_job.create_job(
        model="saaras:v4", mode="transcribe", language_code=language,
        with_diarization=True, with_timestamps=True,
    )
    job.upload_files(file_paths=[str(f) for f in files])
    job.start()
    job.wait_until_complete()
    job.download_outputs(output_dir=str(outdir))
    return job


def main():
    client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
    out = ROOT / "data/real/ref"
    out.mkdir(parents=True, exist_ok=True)
    by_lang = {}
    for uri, (name, lang) in CLIPS.items():
        by_lang.setdefault(lang, []).append(uri)
    for lang, uris in by_lang.items():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            files = []
            for uri in uris:  # short upload names so outputs map back to the uri
                dst = tmp / f"{uri}.mp3"
                shutil.copy(ROOT / "realaudio" / CLIPS[uri][0], dst)
                files.append(dst)
            dl = tmp / "out"
            dl.mkdir()
            run_job(client, files, lang, dl)
            got = sorted(dl.rglob("*.json"))
            print(lang, "outputs:", [g.name for g in got], file=sys.stderr)
            for g in got:
                uri = next((u for u in uris if u in g.stem), None)
                if uri is None:
                    print("unmatched output", g, file=sys.stderr)
                    continue
                shutil.copy(g, out / f"{uri}.sarvam.json")
    manifest = [{"uri": u, "audio": str(Path("realaudio") / n), "language": l[:2],
                 "sarvam": str(Path("data/real/ref") / f"{u}.sarvam.json")} for u, (n, l) in CLIPS.items()]
    (ROOT / "data/real/manifest.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest))


if __name__ == "__main__":
    main()
