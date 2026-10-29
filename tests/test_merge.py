"""CPU-only tests for the ASR/diarization merge logic and metrics (no models needed)."""

import numpy as np

from diarai.metrics import cpwer, normalize, wer
from diarai.pipeline import DiarAI, Turn, Word


def W(s, e, t="x"):
    return Word(s, e, t, 1.0)


def test_assign_by_max_overlap():
    turns = [Turn("A", 0.0, 2.0), Turn("B", 1.8, 4.0)]
    words = [W(0.1, 0.5), W(1.7, 2.0), W(1.9, 2.5), W(3.0, 3.5)]
    DiarAI.assign_speakers(words, turns, smooth=False)
    assert [w.speaker for w in words] == ["A", "A", "B", "B"]


def test_word_in_gap_takes_nearest_turn():
    turns = [Turn("A", 0.0, 1.0), Turn("B", 5.0, 6.0)]
    words = [W(1.2, 1.4), W(4.5, 4.8)]
    DiarAI.assign_speakers(words, turns, smooth=False)
    assert [w.speaker for w in words] == ["A", "B"]


def test_no_turns_single_speaker():
    words = [W(0, 1)]
    DiarAI.assign_speakers(words, [], smooth=False)
    assert words[0].speaker == "SPEAKER_00"


def test_smoothing_fixes_boundary_flip_but_keeps_real_interjection():
    words = [W(0.0, 0.3), W(0.3, 0.6), W(0.65, 0.8), W(0.85, 1.1), W(1.1, 1.4)]
    for w, s in zip(words, "AABAA"):
        w.speaker = s
    DiarAI._smooth(words)
    assert "".join(w.speaker for w in words) == "AAAAA"

    # an interjection separated by pauses is real and must survive
    words = [W(0.0, 0.3), W(0.3, 0.6), W(1.5, 1.8), W(3.0, 3.3), W(3.3, 3.6)]
    for w, s in zip(words, "AABAA"):
        w.speaker = s
    DiarAI._smooth(words)
    assert "".join(w.speaker for w in words) == "AABAA"


def test_group_into_utterances():
    words = [W(0, 0.5, "hello"), W(0.5, 1.0, "there"), W(1.2, 1.5, "hi"), W(5.0, 5.5, "again")]
    for w, s in zip(words, "AABB"):
        w.speaker = s
    utts = DiarAI.group(words, max_gap=1.5)
    assert [(u.speaker, u.text) for u in utts] == [("A", "hello there"), ("B", "hi"), ("B", "again")]


def test_cpwer_is_permutation_invariant():
    ref = {"alice": "hello how are you".split(), "bob": "fine thanks".split()}
    hyp = {"S1": "fine thanks".split(), "S0": "hello how are you".split()}
    errors, n, mapping = cpwer(ref, hyp)
    assert errors == 0 and n == 6 and mapping == {"S1": "bob", "S0": "alice"}


def test_cpwer_counts_misattributed_words():
    ref = {"a": "one two three".split(), "b": "four five".split()}
    hyp = {"x": "one two".split(), "y": "three four five".split()}
    errors, n, _ = cpwer(ref, hyp)
    assert errors == 2 and n == 5  # "three" deleted from a, inserted into b


def test_cpwer_extra_hyp_speaker():
    errors, n, _ = cpwer({"a": ["hi"]}, {"x": ["hi"], "y": ["oops"]})
    assert errors == 1 and n == 1


def test_wer_and_normalizer():
    assert normalize("Hello, World! It's 5 o'clock.") == normalize("hello world it is 5 o'clock")
    assert wer("a b c".split(), "a x c".split()) == (1, 3)


def test_merge_to_max_speakers():
    hard = np.array([[0, 1], [2, 1], [0, 2]])
    cents = np.array([[1.0, 0.0], [0.0, 1.0], [0.99, 0.1]])  # cluster 2 ~ cluster 0
    out = DiarAI._merge_to_max(hard.copy(), cents, 2)
    assert set(np.unique(out)) == {0, 1}
    assert out[1, 0] == out[0, 0]  # 2 merged into 0


def test_overlap_prefers_speaker_who_just_took_the_floor():
    turns = [Turn("A", 0.0, 5.0), Turn("B", 3.0, 8.0)]
    words = [W(1.0, 1.5), W(3.5, 4.0), W(6.0, 6.5)]
    DiarAI.assign_speakers(words, turns, smooth=False)
    assert [w.speaker for w in words] == ["A", "B", "B"]
