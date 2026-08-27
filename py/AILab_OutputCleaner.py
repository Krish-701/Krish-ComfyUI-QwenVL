import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OutputCleanConfig:
    mode: str = "prompt"
    strip_think: bool = True
    strip_code_fences: bool = True
    strip_role_prefixes: bool = True
    strip_json_wrappers: bool = True
    strip_leading_preamble: bool = True
    strip_planning: bool = True
    keep_first_paragraph_only: bool = False


_ROLE_PREFIX_RE = re.compile(r"^\s*(assistant|final|output|response|result|prompt)\s*:\s*", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^\s*```[\w-]*\s*$", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(
    r"<(?:think|thinking)[^>]*>.*?</(?:think|thinking)\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
_THINK_OPEN_RE = re.compile(r"<(?:think|thinking)[^>]*>", flags=re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</(?:think|thinking)\s*>", flags=re.IGNORECASE)
_ALT_THINK_BLOCK_RE = re.compile(
    r"(?:<\|think\|>|◁think▷)(.*?)(?:<\|/think\|>|◁/think▷)",
    flags=re.IGNORECASE | re.DOTALL,
)
_FINAL_MARKER_RE = re.compile(
    r"(?im)^\s*(?:final\s+answer|final|answer|output|response)\s*[:\-]\s*",
)
_MARKER_RE = re.compile(
    r"(?im)^\s*(final|final answer|answer|output|result|prompt)\s*[:\-]\s*",
)
_IM_TOKEN_RE = re.compile(
    r"(?i)<\|?im_(start|end)\|?>|<im_(start|end)>|<\|endoftext\|>",
)
_PLANNING_RE = re.compile(
    r"(?is)\b("
    r"i\s+(should|need|must|will|want|am\s+going\s+to|have\s+to)\b|"
    r"let's\b|"
    r"first\b|next\b|then\b|"
    r"wait\b|"
    r"so\s+i\s+need\s+to\b|"
    r"i\s+should\s+focus\s+on\b"
    r")"
)


def _dedupe_parts(parts: list[str]) -> str:
    seen = set()
    unique = []
    for p in parts:
        t = (p or "").strip()
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return "\n\n".join(unique).strip()


def extract_reasoning_content(text: str, extra: str | None = None) -> str:
    """Pull model reasoning from think tags and optional API reasoning_content."""
    parts: list[str] = []
    extra_text = (extra or "").strip()
    if extra_text:
        parts.append(extra_text)
    raw = text or ""
    for match in re.finditer(
        r"<(?:think|thinking)[^>]*>(.*?)</(?:think|thinking)\s*>",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        inner = (match.group(1) or "").strip()
        if inner:
            parts.append(inner)
    for match in _ALT_THINK_BLOCK_RE.finditer(raw):
        inner = (match.group(1) or "").strip()
        if inner:
            parts.append(inner)
    if not any(p != extra_text for p in parts):
        open_m = _THINK_OPEN_RE.search(raw)
        if open_m:
            after = raw[open_m.end() :]
            close_m = _THINK_CLOSE_RE.search(after)
            body = after[: close_m.start()] if close_m else after
            marker = _FINAL_MARKER_RE.search(body)
            if marker and not close_m:
                body = body[: marker.start()]
            body = body.strip()
            if body:
                parts.append(body)
    return _dedupe_parts(parts)


def _strip_think_markup(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    cleaned = _ALT_THINK_BLOCK_RE.sub("", cleaned)
    cleaned = _THINK_CLOSE_RE.sub("", cleaned)
    open_m = _THINK_OPEN_RE.search(cleaned)
    if open_m:
        after = cleaned[open_m.end() :]
        marker = _FINAL_MARKER_RE.search(after)
        cleaned = after[marker.end() :] if marker else ""
    return cleaned.strip()


def _strip_reasoning_overlap(response: str, reasoning: str) -> str:
    resp = (response or "").strip()
    reason = (reasoning or "").strip()
    if not resp:
        return ""
    if reason and resp.startswith(reason):
        resp = resp[len(reason) :].strip()
    if reason and reason in resp:
        # Drop a leading copy of the think block if the model echoed it.
        idx = resp.find(reason)
        if idx == 0:
            resp = resp[len(reason) :].strip()
    return resp.strip()


def split_response_and_reasoning(text: str, extra_reasoning: str | None = None, mode: str = "text") -> tuple[str, str]:
    raw = text or ""
    extra = (extra_reasoning or "").strip()
    reasoning = extract_reasoning_content(raw, extra)

    answer = raw
    close_matches = list(_THINK_CLOSE_RE.finditer(raw))
    if close_matches:
        answer = raw[close_matches[-1].end() :]
    else:
        alt_matches = list(_ALT_THINK_BLOCK_RE.finditer(raw))
        if alt_matches:
            answer = raw[alt_matches[-1].end() :]
        else:
            answer = _strip_think_markup(raw)
            if extra and extra in answer:
                answer = answer.replace(extra, "", 1)

    answer = _strip_think_markup(answer)
    answer = _IM_TOKEN_RE.sub("", answer).strip()
    answer = _strip_reasoning_overlap(answer, reasoning)

    if extra and not reasoning:
        reasoning = extra
        answer = _strip_reasoning_overlap(answer, extra)

    if answer and reasoning and answer == reasoning:
        answer = ""

    # Light cleanup on the answer only — never keep think text in RESPONSE.
    if answer:
        answer = clean_model_output(
            answer,
            OutputCleanConfig(
                mode=mode,
                strip_think=True,
                strip_planning=False,
                strip_leading_preamble=False,
            ),
        )
        answer = _strip_reasoning_overlap(answer, reasoning)

    return answer.strip(), reasoning.strip()


def clean_model_output(text: str, config: OutputCleanConfig | None = None) -> str:
    if not text:
        return ""

    cfg = config or OutputCleanConfig()
    cleaned = (text or "").strip()

    # Remove common chat template tokens that sometimes leak into output.
    cleaned = _IM_TOKEN_RE.sub("", cleaned).strip()

    if cfg.strip_think:
        cleaned = _THINK_BLOCK_RE.sub("", cleaned)
        cleaned = _THINK_CLOSE_RE.sub("", cleaned)
        if _THINK_OPEN_RE.search(cleaned):
            cleaned = _THINK_OPEN_RE.sub("", cleaned)
            parts = re.split(r"\n\s*\n", cleaned, maxsplit=1)
            if len(parts) == 2:
                cleaned = parts[1]
        cleaned = cleaned.strip()

    cleaned = _IM_TOKEN_RE.sub("", cleaned).strip()

    if cfg.strip_code_fences and "```" in cleaned:
        lines = [ln for ln in cleaned.splitlines() if not _CODE_FENCE_RE.match(ln)]
        cleaned = "\n".join(lines).strip()

    if cfg.strip_json_wrappers:
        maybe = _extract_from_json(cleaned, mode=cfg.mode)
        if maybe is not None:
            cleaned = maybe.strip()

    if cfg.strip_leading_preamble:
        cleaned = _drop_preamble(cleaned).strip()

    if cfg.strip_planning and cfg.mode == "prompt":
        without_planning = _strip_planning_paragraphs(cleaned)
        if without_planning:
            cleaned = without_planning

    if cfg.strip_role_prefixes:
        lines = cleaned.splitlines()
        if lines:
            lines[0] = _ROLE_PREFIX_RE.sub("", lines[0])
        cleaned = "\n".join(lines).strip()

    cleaned = _MARKER_RE.sub("", cleaned).strip()

    if cfg.keep_first_paragraph_only:
        parts = re.split(r"\n\s*\n", cleaned, maxsplit=1)
        cleaned = parts[0].strip()

    return cleaned


def _extract_from_json(text: str, mode: str) -> str | None:
    candidate = text.strip()
    if not candidate:
        return None

    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None

    try:
        payload = json.loads(candidate)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    preferred_keys = []
    if mode == "prompt":
        preferred_keys = ["prompt", "final", "output", "text", "content"]
    else:
        preferred_keys = ["text", "content", "output", "final"]

    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return None


def _drop_preamble(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    keep_from = 0
    for i, ln in enumerate(lines[:20]):
        if _MARKER_RE.match(ln):
            keep_from = i
        if re.search(r"(?i)\bhere(?:'s| is)\b", ln) and i < 6:
            keep_from = max(keep_from, i + 1)

    return "\n".join(lines[keep_from:]).strip()


def _strip_planning_paragraphs(text: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]
    if not paragraphs:
        return ""

    kept: list[str] = []
    dropping = True
    for p in paragraphs:
        is_planning = bool(_PLANNING_RE.search(p))
        if dropping and is_planning:
            continue
        dropping = False
        kept.append(p)

    # If everything looked like planning, don't destroy the output.
    if not kept:
        return text.strip()
    return "\n\n".join(kept).strip()
