"""
processors/ack_classifier.py — Tells a backchannel ("uh huh") from a real utterance.

Words like "yes"/"sure" count as backchannel only when the agent wasn't asking anything.
"""

from __future__ import annotations

import re
from enum import Enum

from config import ACK_CONTEXT_WORDS, ACK_PHRASES, ACK_SINGLE_WORDS

_NORM_RE  = re.compile(r"[\-]")
_STRIP_RE = re.compile(r"[^\w\s]")


class InterruptType(Enum):
    NONE           = 0
    ACKNOWLEDGEMENT = 1
    SUBSTANTIVE    = 2


def normalize(text: str) -> str:
    """Lowercase, collapse hyphens to spaces, strip punctuation."""
    t = _NORM_RE.sub(" ", text.lower())
    t = _STRIP_RE.sub("", t)
    return " ".join(t.split())


def strip_leading_dash(text: str) -> str:
    """Remove leading '- ' or '-' prefix that Whisper sometimes adds."""
    t = text.strip()
    if t.startswith("- "):
        return t[2:].strip()
    if t.startswith("-"):
        return t[1:].strip()
    return t


def is_annotation(text: str) -> bool:
    """Return True if text is purely a non-speech annotation (cough, [BLANK_AUDIO], etc.)."""
    cleaned = re.sub(r"\([^)]*\)", "", text)
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"\*+([^*]+)\*+", r"\1", cleaned)
    cleaned = strip_leading_dash(cleaned)
    if not cleaned.strip():
        return True
    non_speech = {
        "cough", "coughs", "coughing", "coughed", "sigh", "sighs", "sighing",
        "laugh", "laughter", "laughs", "breath", "breathing", "inhale",
        "exhale", "sniff", "ahem", "clears throat", "clears", "throat",
    }
    return normalize(cleaned) in non_speech


def classify_interrupt(text: str, agent_asked_question: bool = False) -> InterruptType:
    """Classify text as ACKNOWLEDGEMENT, SUBSTANTIVE or NONE.

    When the agent just asked a question, "yes"/"no" are answers, not backchannels.
    """
    norm = normalize(strip_leading_dash(text))
    if not norm:
        return InterruptType.NONE
    if norm in ACK_PHRASES:
        return InterruptType.ACKNOWLEDGEMENT

    words = norm.split()
    if len(words) == 1:
        w = words[0]
        if w in ACK_SINGLE_WORDS:
            return InterruptType.ACKNOWLEDGEMENT
        if w in ACK_CONTEXT_WORDS and not agent_asked_question:
            return InterruptType.ACKNOWLEDGEMENT
        return InterruptType.SUBSTANTIVE

    # Several words only count as ACK if every one is unambiguous filler.
    if all(w in ACK_SINGLE_WORDS for w in words):
        return InterruptType.ACKNOWLEDGEMENT
    return InterruptType.SUBSTANTIVE


def jaccard(a: str, b: str, norm_re: re.Pattern = re.compile(r"[^\w\s]")) -> float:
    """Word-level Jaccard similarity between two strings."""
    if not a or not b:
        return 0.0
    na = " ".join(norm_re.sub("", a.lower().strip()).split())
    nb = " ".join(norm_re.sub("", b.lower().strip()).split())
    if na == nb:
        return 1.0
    w1, w2 = set(na.split()), set(nb.split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)