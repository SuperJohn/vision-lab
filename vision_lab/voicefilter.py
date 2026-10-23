"""Junk filter for recording-mode Whisper transcripts (voice log garbage).

Whisper invents text on silence and room noise: stray symbols or odd Unicode ("Ƣ"), lone letters,
a lone "the", "Thank you." and the like. is_junk_transcript() is a pure function used in two places:
  - vision_lab/voicetx.py drops junk segments before insert, so they are never saved;
  - vision_lab/voicelog.py and vision_lab/captures.py hide old junk rows at display time
    (the rows stay in vlab.db untouched).
Only counts of dropped segments are logged, never their text.

Rules, in order (the first that matches wins):
  empty         nothing left after trimming spaces and punctuation
  no_latin      no ASCII/Latin-1 letters or digits (stray symbols, "Ƣ", "ƥ", other odd Unicode)
  (keep)        whole text is a real short utterance on the ALLOW list (yes, no, ok, hi, stop, ...)
  short         1-2 letters/digits in total
  stopword      a lone filler or stopword (the, a, uh, um, you, ...)
  hallucination a known Whisper hallucination phrase as the whole segment ("Thank you.", ...)
  repeated      mostly one repeated character or word
  no_speech     meta: no_speech_prob > 0.6 and avg_logprob < -1.0
  compression   meta: compression_ratio > 2.4 (Whisper's own repetition-loop signal)
Names and normal sentences pass, since none of the rules match them."""
import re
from collections import Counter

MAX_NO_SPEECH, MIN_LOGPROB, MAX_COMPRESSION = 0.6, -1.0, 2.4

ALLOW = frozenset("""yes no ok okay hi hey hello bye stop go wait yeah yep yup nope sure right cool nice
done start help what why who how where when wow great fine good""".split())

STOPWORDS = frozenset("""the a an t s i o and or but so of to in on at it is be as by if for from that this
then than uh um umm uhm uh-huh hmm hm mm mmm mhm huh ah oh eh er erm you your me my we he she they""".split())

HALLUCINATIONS = frozenset(s.strip() for s in """
thank you | thank you very much | thank you so much | thanks for watching | thank you for watching
thanks for watching! | thank you so much for watching | thanks for listening | thank you for listening
please subscribe | subscribe to my channel | like and subscribe | don't forget to subscribe
see you next time | see you in the next video | i'll see you in the next video | bye bye
subtitles by the amara.org community | subtitles by the amara org community | transcribed by
transcription by castingwords | you you
""".replace("\n", "|").split("|") if s.strip())

_LATIN = re.compile(r"[A-Za-z0-9\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF]")
_EDGE = re.compile(r"^[\W_]+|[\W_]+$")
_SPACE = re.compile(r"\s+")


def normalize(text):
    """Lowercase, collapse spaces, trim punctuation at both ends."""
    t = _SPACE.sub(" ", (text or "").strip().lower())
    return _EDGE.sub("", t).strip()


def junk_reason(text, meta=None):
    """Why a transcript segment is junk, or None if it should be kept. Pure, no I/O."""
    core = normalize(text)
    if not core:
        return "empty"
    latin = _LATIN.findall(core)
    if not latin:
        return "no_latin"
    if core in ALLOW:
        return None
    if len(latin) <= 2:
        return "short"
    if core in STOPWORDS:
        return "stopword"
    if core in HALLUCINATIONS:
        return "hallucination"
    words = [_EDGE.sub("", w) for w in core.split(" ")]
    words = [w for w in words if w]
    if len(words) >= 3:
        top, n = Counter(words).most_common(1)[0]
        if (n == len(words) and top not in ALLOW) or (len(words) >= 4 and n / len(words) >= 0.75):
            return "repeated"
    chars = [c for c in latin]
    if len(chars) >= 4 and Counter(chars).most_common(1)[0][1] / len(chars) >= 0.8:
        return "repeated"
    m = meta or {}
    nsp, lp, cr = m.get("no_speech_prob"), m.get("avg_logprob"), m.get("compression_ratio")
    if nsp is not None and lp is not None and nsp > MAX_NO_SPEECH and lp < MIN_LOGPROB:
        return "no_speech"
    if cr is not None and cr > MAX_COMPRESSION:
        return "compression"
    return None


def is_junk_transcript(text, meta=None):
    """True when a Whisper segment is noise/hallucination and should not be saved or shown."""
    return junk_reason(text, meta) is not None
