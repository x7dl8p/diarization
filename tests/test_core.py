"""CPU-only tests (no models needed): pipeline helpers, metrics, formats, auth, streaming logic."""

import base64
import io
import threading
import time
import wave

import numpy as np
import pytest

from diarai.audio import decode_chunk
from diarai.eval.metrics import cpwer, normalize, wer
from diarai.formats import lang_code, parse_lang, sarvam_response
from diarai.models import PriorityLock
from diarai.pipeline import DiarAI, Result, Turn, Utterance, Word, dedupe_crosstalk
from diarai.server.auth import Authenticator, extract_key, generate_key, hash_key, verify_key
from diarai.stream import HOP, Endpointer, OnlineSpeakers, Timeline, runs


def W(s, e, t="x", spk=None):
    return Word(s, e, t, 1.0, spk)


# ------------------------------------------------------------------ pipeline
def test_group_into_utterances():
    words = [W(0, 0.5, "hello"), W(0.5, 1.0, "there"), W(1.2, 1.5, "hi"), W(5.0, 5.5, "again")]
    for w, s in zip(words, "AABB"):
        w.speaker = s
    utts = DiarAI.group(words, max_gap=1.5)
    assert [(u.speaker, u.text) for u in utts] == [("A", "hello there"), ("B", "hi"), ("B", "again")]


def test_group_keeps_overlapped_speech_together_but_splits_real_turns():
    A = [W(0.0, 0.5, "staff", "A"), W(0.5, 1.0, "overlap", "A"), W(1.2, 1.5, "terms", "A"), W(1.6, 2.0, "behavior.", "A")]
    B = [W(1.0, 1.2, "thumbs", "B"), W(1.5, 1.6, "up", "B")]
    utts = DiarAI.group(sorted(A + B, key=lambda w: w.start))
    assert [(u.speaker, u.text) for u in utts] == [("A", "staff overlap terms behavior."), ("B", "thumbs up")]
    # a real reply (>= 1 s) in between ends A's turn even if A resumes within max_gap
    words = [W(0, 1, "question", "A"), W(1.1, 2.3, "answer", "B"), W(2.4, 3, "again", "A")]
    assert [u.text for u in DiarAI.group(words)] == ["question", "answer", "again"]


def test_group_splits_long_monologue_at_sentence_end():
    words = [W(i, i + 0.9, f"w{i}" + ("." if i == 21 else ""), "A") for i in range(30)]
    utts = DiarAI.group(words, max_len=20.0)
    assert [len(u.words) for u in utts] == [22, 8]


def test_speaker_segments_merge_short_pauses_and_split_long():
    turns = [Turn("A", 0, 2), Turn("A", 2.3, 4), Turn("B", 4, 5), Turn("A", 10, 70)]
    units = DiarAI.speaker_segments(turns, max_gap=0.6, max_len=25.0)
    assert units[0] == ("A", 0, 4)
    assert units[1] == ("B", 4, 5)
    long = [u for u in units if u[1] >= 10]
    assert len(long) == 3 and all(b - a <= 25 for _, a, b in long)


def test_merge_to_max_speakers():
    hard = np.array([[0, 1], [2, 1], [0, 2]])
    cents = np.array([[1.0, 0.0], [0.0, 1.0], [0.99, 0.1]])  # cluster 2 ~ cluster 0
    out = DiarAI._merge_to_max(hard.copy(), cents, 2)
    assert set(np.unique(out)) == {0, 1}
    assert out[1, 0] == out[0, 0]  # 2 merged into 0


def test_crosstalk_duplicate_kept_in_longer_unit_only():
    long = [W(1.0, 1.3, "yes", "A"), W(1.4, 1.8, "sir", "A")]
    short = [W(1.05, 1.3, "Yes.", "B")]
    words = dedupe_crosstalk([(5.0, long), (0.5, short)])
    assert [(w.text, w.speaker) for w in words] == [("yes", "A"), ("sir", "A")]


# ------------------------------------------------------------------- metrics
def test_cpwer_is_permutation_invariant():
    ref = {"alice": "hello how are you".split(), "bob": "fine thanks".split()}
    hyp = {"S1": "fine thanks".split(), "S0": "hello how are you".split()}
    errors, n, mapping = cpwer(ref, hyp)
    assert errors == 0 and n == 6 and mapping == {"S1": "bob", "S0": "alice"}


def test_cpwer_counts_misattributed_words():
    ref = {"a": "one two three".split(), "b": "four five".split()}
    hyp = {"x": "one two".split(), "y": "three four five".split()}
    errors, n, _ = cpwer(ref, hyp)
    assert errors == 2 and n == 5


def test_wer_and_normalizer():
    assert normalize("Hello, World! It's 5 o'clock.") == normalize("hello world it is 5 o'clock")
    assert wer("a b c".split(), "a x c".split()) == (1, 3)


# ------------------------------------------------------------------- formats
def test_sarvam_response_shape():
    utts = [Utterance("SPEAKER_03", 0.0, 1.0, "hello there", [W(0, 0.5, "hello"), W(0.5, 1.0, "there")]),
            Utterance("SPEAKER_01", 1.2, 2.0, "hi", [W(1.2, 2.0, "hi")])]
    r = Result(utts, [], "hi", 2.0, {"total": 0.5}, 0.97)
    out = sarvam_response(r, "rid")
    assert set(out) >= {"request_id", "transcript", "timestamps", "diarized_transcript", "language_code",
                        "language_probability"}
    assert out["transcript"] == "hello there hi" and out["language_code"] == "hi-IN"
    assert [e["speaker_id"] for e in out["diarized_transcript"]["entries"]] == ["0", "1"]
    assert out["timestamps"]["words"] == ["hello", "there", "hi"]
    assert sarvam_response(r, "x", with_timestamps=False, with_diarization=False)["timestamps"] is None


def test_language_codes():
    assert parse_lang("hi-IN") == "hi" and parse_lang("unknown") is None and parse_lang(None) is None
    assert parse_lang("od-IN") == "or" and lang_code("or") == "od-IN" and lang_code("en") == "en-IN"
    assert lang_code("fr") == "fr"


# ---------------------------------------------------------------------- auth
def test_key_hash_roundtrip():
    key = generate_key()
    assert len(key) >= 64
    h = hash_key(key)
    assert key not in h and h.startswith("scrypt$")
    assert verify_key(key, h) and not verify_key(key + "x", h) and not verify_key(key, "garbage")
    assert hash_key(key) != h  # salted


def test_extract_key_from_headers():
    assert extract_key({"api-subscription-key": " k1 "}) == "k1"
    assert extract_key({"authorization": "Bearer k2"}) == "k2"
    basic = base64.b64encode(b"anyone:k3").decode()
    assert extract_key({"authorization": f"Basic {basic}"}) == "k3"
    assert extract_key({"authorization": "Basic !!notbase64"}) is None
    assert extract_key({}) is None


def test_authenticator_caches_and_throttles():
    key = generate_key()
    auth = Authenticator([hash_key(key)], max_failures=3)
    assert auth.check(key, "c") and auth.check(key, "c")
    for _ in range(3):
        assert not auth.check("wrong", "evil")
    assert auth.throttled("evil") and not auth.throttled("c")
    assert not auth.check(None, "x") and not auth.check("k" * 1000, "x")


# ----------------------------------------------------------------- streaming
def _wav(x, sr):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def test_decode_chunk_wav_pcm_and_resample():
    x = (0.5 * np.sin(np.arange(1600) / 5)).astype(np.float32)
    assert np.allclose(decode_chunk(_wav(x, 16000)), x, atol=1e-3)
    pcm = (x * 32767).astype("<i2").tobytes()
    assert np.allclose(decode_chunk(pcm, 16000, "pcm_s16le"), x, atol=1e-3)
    assert abs(len(decode_chunk(_wav(np.zeros(2400, np.float32), 24000))) - 1600) <= 1
    with pytest.raises(ValueError):
        decode_chunk(b"\x00\x00", 16000, "mp3")


def _frames(seconds_speech_pattern):
    probs = []
    for speech, sec in seconds_speech_pattern:
        probs += [0.9 if speech else 0.05] * round(sec / HOP)
    return np.array(probs)


def test_endpointer_segments_on_silence():
    ep = Endpointer(end_silence=0.6, max_utterance=20)
    ev = ep.push(_frames([(False, 1.0), (True, 2.0), (False, 0.3), (True, 1.0), (False, 1.0), (True, 0.5)]))
    kinds = [e.kind for e in ev]
    assert kinds == ["start", "segment", "end", "start"]  # 0.3 s pause does not split
    seg = ev[1]
    assert 0.6 < seg.start < 1.0 and 4.2 < seg.end < 4.7
    ev = ep.flush()
    assert [e.kind for e in ev] == ["segment", "end"] and ev[0].start >= seg.end


def test_endpointer_cuts_long_speech_at_quietest_point():
    ep = Endpointer(end_silence=0.6, max_utterance=5.0)
    p = _frames([(True, 4.0)])
    p = np.r_[p, [0.4] * 3, _frames([(True, 4.0)])]  # a dip (still speech) around 4 s
    segs = [e for e in ep.push(p) if e.kind == "segment"]
    assert len(segs) == 1 and abs(segs[0].end - 4.05) < 0.1


def test_online_speakers():
    a, b = np.eye(4)[0], np.eye(4)[1]
    spk = OnlineSpeakers(threshold=0.3, min_new_dur=2.0)
    assert spk.assign(a, 3.0) == 0
    assert spk.assign(a * 0.9 + b * 0.1, 3.0) == 0
    assert spk.assign(b, 1.0) == 0  # too short to open a new speaker
    assert spk.assign(b, 3.0) == 1
    assert spk.assign(a, 3.0, exclude={0}) == 0  # same voice split by EEND: no new speaker
    assert len(spk) == 2


def test_timeline_and_runs():
    tl = Timeline()
    tl.mark(1, 1.0, 2.0)
    assert list(tl.at(1, np.array([0.5, 1.5, 2.5, 999.0]))) == [False, True, False, False]
    assert not tl.at(0, np.array([1.5])).any()
    assert runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]


def test_priority_lock_serves_high_priority_first():
    lock, order = PriorityLock(), []
    lock_held = threading.Event()

    def holder():
        with lock(2):
            lock_held.set()
            time.sleep(0.2)

    def worker(p, name):
        with lock(p):
            order.append(name)

    t0 = threading.Thread(target=holder)
    t0.start()
    lock_held.wait()
    ts = [threading.Thread(target=worker, args=(2, "batch"))]
    ts[0].start()
    time.sleep(0.05)
    ts.append(threading.Thread(target=worker, args=(0, "stream")))
    ts[1].start()
    for t in [t0] + ts:
        t.join()
    assert order == ["stream", "batch"]
    with lock(1):
        with lock(1):  # re-entrant
            pass


def test_maya_asr_uses_speaker_pieces(monkeypatch):
    """asr="maya": the same diarization pieces go to the maya service in batches; one segment Word per piece,
    junk (too many words for the piece's length) dropped."""
    from contextlib import nullcontext

    import diarai.pipeline as P

    sent = []

    def fake(clips, **kw):
        sent.append([len(c) for c in clips])
        return ["नमस्ते सर" if len(c) > 16000 * 2 else "एक दो तीन चार पांच छह सात आठ नौ दस" for c in clips]

    monkeypatch.setattr(P, "maya_transcribe", fake)
    d = P.DiarAI.__new__(P.DiarAI)
    d.models = type("M", (), {"gpu": lambda self, p: nullcontext()})()
    d.asr = "maya"
    wav = np.zeros(16000 * 40, dtype=np.float32)
    words = d.transcribe_turns(wav, [P.Turn("0", 0, 5), P.Turn("1", 5.2, 9), P.Turn("0", 9.5, 10.0)], "hi")
    assert len(sent) == 1 and len(sent[0]) == 3
    assert [(w.speaker, w.start, w.end, w.text) for w in words] == [("0", 0, 5, "नमस्ते सर"), ("1", 5.2, 9, "नमस्ते सर")]
