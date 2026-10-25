"""Read transcripts (.srt / .vtt / .json / .txt) and cut them into fixed time windows."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    index: int
    start: float
    end: float
    text: str


# --------------------------------------------------------------------------- helpers
_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?")
_CUE = re.compile(r"^\s*([\d:.,]+)\s*-->\s*([\d:.,]+)")
_TAG = re.compile(r"<[^>]+>")
_TXT_LINE = re.compile(
    r"^\s*[\[\(]?(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)[\]\)]?\s*[-–—:]?\s*(.*)$"
)


def parse_ts(value: str) -> float:
    m = _TS.fullmatch(value.strip())
    if not m:
        raise ValueError(f"Bad timestamp: {value!r}")
    h, mi, s, ms = m.groups()
    frac = int(ms.ljust(3, "0")) / 1000 if ms else 0.0
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + frac


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _TAG.sub("", text)).strip()


# --------------------------------------------------------------------------- parsers
def _load_cues(raw: str) -> list[Segment]:
    """SRT and VTT share the 'start --> end' cue layout."""
    lines = raw.splitlines()
    segs: list[Segment] = []
    i = 0
    while i < len(lines):
        m = _CUE.match(lines[i])
        if not m:
            i += 1
            continue
        start, end = parse_ts(m.group(1)), parse_ts(m.group(2))
        i += 1
        buf = []
        while i < len(lines) and lines[i].strip():
            buf.append(lines[i])
            i += 1
        text = _clean(" ".join(buf))
        if text:
            segs.append(Segment(start, end, text))
    return segs


def _load_json(raw: str) -> list[Segment]:
    """Handles youtube-transcript-api output ({text,start,duration}) and Whisper ({text,start,end})."""
    data = json.loads(raw)
    if isinstance(data, dict):
        data = (
            data.get("segments")
            or data.get("transcript")
            or data.get("snippets")
            or data.get("results")
            or []
        )
    segs: list[Segment] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = _clean(str(item.get("text", "")))
        if not text:
            continue
        start = float(item.get("start", item.get("offset", 0)) or 0)
        if "end" in item and item["end"] is not None:
            end = float(item["end"])
        else:
            end = start + float(item.get("duration", 0) or 0)
        segs.append(Segment(start, end, text))
    return segs


def _split_long(sentence: str, max_words: int = 30) -> list[str]:
    words = sentence.split()
    if len(words) <= max_words + 10:
        return [sentence]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _spread_untimed(text: str, duration: float | None) -> list[Segment]:
    """No timestamps at all: spread sentences across the video length proportionally."""
    sentences: list[str] = []
    for s in re.split(r"(?<=[.!?।॥])\s+", text):
        s = s.strip()
        if s:
            sentences.extend(_split_long(s))
    if not sentences:
        return []
    if not duration:
        duration = max(len(text.split()) / 2.5, 1.0)  # ~150 words/min fallback
    total_chars = sum(len(s) for s in sentences) or 1
    segs, cursor = [], 0.0
    for s in sentences:
        span = duration * len(s) / total_chars
        segs.append(Segment(cursor, cursor + span, s))
        cursor += span
    return segs


def _load_txt(raw: str, duration: float | None) -> list[Segment]:
    """Plain text. Supports '[mm:ss] text' / 'mm:ss text' lines; otherwise untimed."""
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    matched = sum(1 for ln in lines if _TXT_LINE.match(ln))
    if lines and matched >= 0.5 * len(lines):
        entries: list[list] = []  # [start, text]
        for ln in lines:
            m = _TXT_LINE.match(ln)
            if m:
                entries.append([parse_ts(m.group(1)), m.group(2).strip()])
            elif entries:
                entries[-1][1] += " " + ln.strip()
        segs = []
        for k, (start, text) in enumerate(entries):
            if k + 1 < len(entries):
                end = entries[k + 1][0]
            else:
                end = duration if duration and duration > start else start + 5
            text = _clean(text)
            if text:
                segs.append(Segment(start, end, text))
        return segs
    return _spread_untimed(" ".join(ln.strip() for ln in lines), duration)


def load_transcript(path: Path, duration: float | None = None) -> list[Segment]:
    """Parse any supported transcript file into time-stamped segments.

    `duration` (seconds, e.g. from the audio file) is only used when the
    transcript carries no timestamps.
    """
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    suffix = path.suffix.lower()
    caption_like = True  # caption/JSON files may contain rolling duplicate lines
    if suffix in {".srt", ".vtt"}:
        segs = _load_cues(raw)
    elif suffix == ".json":
        segs = _load_json(raw)
    elif "-->" in raw:
        segs = _load_cues(raw)
    else:
        segs = _load_txt(raw, duration)
        caption_like = False

    segs.sort(key=lambda s: s.start)
    if not caption_like:  # plain text: repeated sentences are real content, keep them
        return segs

    deduped: list[Segment] = []  # YouTube auto-captions often repeat the previous line
    for s in segs:
        if deduped and deduped[-1].text == s.text:
            deduped[-1].end = max(deduped[-1].end, s.end)
        else:
            deduped.append(s)
    return deduped


# --------------------------------------------------------------------------- chunking
def chunk_segments(segments: list[Segment], window: int = 75, overlap: int = 0) -> list[Chunk]:
    """Group segments into fixed `window`-second blocks (0-75s, 75-150s, ...).

    A segment belongs to the block its start time falls into. With overlap > 0 the
    blocks slide by (window - overlap) seconds, so a segment can appear in two blocks.
    """
    if not segments:
        return []
    step = max(window - overlap, 1)
    total_end = max(s.end for s in segments)
    last_start = max(s.start for s in segments)

    chunks: list[Chunk] = []
    idx, w = 0, 0.0
    while w <= last_start:
        members = [s for s in segments if w <= s.start < w + window]
        if members:
            text = " ".join(s.text for s in members)
            chunks.append(Chunk(index=idx, start=w, end=min(w + window, total_end), text=text))
        idx += 1
        w += step
    return chunks
