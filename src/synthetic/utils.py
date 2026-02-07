"""
Utility functions for text normalization and robust parsing of model outputs.

This module intentionally stays "model-agnostic": it doesn't call the LLM;
it only cleans/normalizes strings and extracts candidate utterances.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
import unicodedata


def normalize(s: str) -> str:
    """
    Normalize an utterance for robust deduplication.

    Parameters
    ----------
    s:
        Input string.

    Returns
    -------
    key:
        Lowercased, whitespace-collapsed, quote-stripped, punctuation-trimmed form.

    Notes
    -----
    This is *not* semantic normalization—it's a stable key for exact-ish dedupe.
    """
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'`]+", "", s)
    s = s.strip(" \t\n\r.,!?;:()[]{}")
    return s


def clean_generated_text(s: str) -> str:
    """
    Clean a single model-generated utterance into human-readable text.

    Parameters
    ----------
    s:
        Raw candidate string produced by the model.

    Returns
    -------
    cleaned:
        A cleaned version suitable for writing to the output file.

    What this fixes
    --------------
    Some models sometimes output:
    - SentencePiece-style word boundary markers: "▁" (U+2581)
    - Underscores used as word separators: "how_would_you_say"
    - Underscore wrappers: "_hello_"
    - Very occasionally, strings with *no* spaces at all: "howwouldyousay..."

    Cleanup steps
    -------------
    1) Replace "▁" markers with spaces.
    2) Strip underscore wrappers.
    3) Convert runs of underscores between word characters into spaces.
    4) Collapse repeated whitespace.
    5) Optional: if still no spaces and mostly alphabetic, try `wordninja` if installed.
    """
    if not s:
        return ""

    t = s.strip()

    # Replace SentencePiece whitespace marker (looks like an underscore bar).
    t = t.replace("▁", " ")

    # Strip surrounding underscore wrappers like "_hello_" or "__hello__".
    t = re.sub(r"^_+", "", t)
    t = re.sub(r"_+$", "", t)

    # Turn underscores that behave like word separators into spaces.
    t = re.sub(r"(?<=\w)_+(?=\w)", " ", t)

    # If the model used underscores instead of spaces everywhere.
    if "_" in t and " " not in t:
        t = t.replace("_", " ")

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Best-effort: recover spaces if the model returned a long single token.
    if t and " " not in t and len(t) >= 18 and re.fullmatch(r"[A-Za-z']+", t):
        try:
            import wordninja  # type: ignore

            parts = wordninja.split(t)
            if len(parts) >= 2:
                t = " ".join(parts)
        except Exception:
            pass

    return t


def is_bad_candidate(s: str) -> bool:
    """
    Heuristics to reject low-quality / corrupted generations.
    """
    if not s:
        return True

    t = s.strip()
    if not t:
        return True

    # reject control chars / combining marks (often from corrupted output)
    for ch in t:
        cat = unicodedata.category(ch)
        if cat.startswith("C") or cat.startswith("M"):
            return True

    # too many non-alnum characters
    alnum = sum(ch.isalnum() for ch in t)
    if alnum / max(1, len(t)) < 0.4:
        return True

    # too many digits (e.g., IDs, timestamps, counters)
    digits = sum(ch.isdigit() for ch in t)
    if digits / max(1, len(t)) > 0.20:
        return True

    # reject UUID-like strings
    if re.search(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", t):
        return True

    # reject long digit runs (IDs / junk)
    if re.search(r"\d{8,}", t):
        return True

    words = t.split()
    if len(words) == 0:
        return True
    if len(words) < 4:
        return True

    # too many single-letter tokens (e.g., "a s d f g h")
    singletons = sum(1 for w in words if len(w) == 1)
    if singletons / max(1, len(words)) >= 0.6:
        return True

    # too many numeric or punctuation-only tokens
    numeric_tokens = sum(1 for w in words if re.fullmatch(r"\d+", w))
    punct_tokens = sum(1 for w in words if re.fullmatch(r"[^A-Za-z0-9]+", w))
    if (numeric_tokens + punct_tokens) / max(1, len(words)) >= 0.4:
        return True

    # reject extremely long tokens (often junk)
    if any(len(w) > 25 for w in words):
        return True

    # hard reject placeholder templates like _A_, __B__, _x_, etc.
    if re.search(r"(?i)(?:^|\\b)_+[a-z]_+(?:\\b|$)", t):
        return True

    # hard reject alphabet/range-like sequences (e.g., "a b c d", "abcd efgh", "A B C ...")
    if re.search(r"(?i)\\b(?:[a-z]\\s+){5,}[a-z]\\b", t):
        return True
    if re.search(r"(?i)\\b(?:abcde|fghij|klmno|pqrst|uvwxy|vwxyz)\\b", t):
        return True
    if re.search(r"(?i)\\b[a-z]{16,}\\b", t):
        return True

    # reject patterns like "a b c d e f g" (many single-letter tokens)
    if singletons / max(1, len(words)) >= 0.4 and len(words) >= 6:
        return True

    # low character diversity (e.g., "SSSSAAAANNNN...")
    letters_only = re.sub(r"[^A-Za-z]", "", t)
    if letters_only:
        uniq_ratio = len(set(letters_only)) / max(1, len(letters_only))
        if uniq_ratio < 0.2:
            return True

    # repeated character junk (e.g., "aaaaaa", "-----")
    if re.fullmatch(r"(.)\1{5,}", t.replace(" ", "")):
        return True

    return False


def strip_think_and_fences(text: str) -> str:
    """
    Remove common wrappers that models sometimes add.

    Parameters
    ----------
    text:
        Raw model output.

    Returns
    -------
    cleaned:
        Text with <think>...</think> blocks removed and markdown fences stripped.

    Why
    ---
    Some models emit chain-of-thought tags or surround JSON in ```json fences.
    Stripping these improves JSON parsing reliability.
    """
    t = text.strip()
    t = re.sub(r"<think>[\s\S]*?</think>", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    return t


def try_parse_json_array(text: str) -> Optional[List[str]]:
    """
    Try to parse a JSON array-of-strings from a model response.

    Handles:
      1) A JSON array of strings: ["a", "b"]
      2) A JSON string containing a JSON array: "[\"a\", \"b\"]"  (double-encoded)
      3) A JSON array embedded inside extra text (we extract the first [...] block)

    Returns
    -------
    arr_or_none:
        A list[str] if a JSON array-of-strings is found, else None.
    """
    t = strip_think_and_fences(text)

    # 1) Parse the whole payload (supports double-encoded via coerce)
    try:
        arr = coerce_json_array_of_strings(t)
        return [x.strip() for x in arr if x and x.strip()]
    except Exception:
        pass

    # 2) If extra text exists, parse the first [...] block (supports double-encoded too)
    m = re.search(r"\[[\s\S]*\]", t)
    if m:
        try:
            arr = coerce_json_array_of_strings(m.group(0))
            return [x.strip() for x in arr if x and x.strip()]
        except Exception:
            pass

    return None


def coerce_json_array_of_strings(text: str):
    """
    Accepts:
      1) a JSON array of strings: ["a", "b"]
      2) a JSON string containing a JSON array: "[\"a\", \"b\"]"
    Returns: list[str]
    """
    obj = json.loads(text)  # first parse

    # Case 2: double-encoded
    if isinstance(obj, str):
        obj = json.loads(obj)  # parse again

    if not (isinstance(obj, list) and all(isinstance(x, str) for x in obj)):
        raise ValueError("Not a JSON array of strings")

    return obj



def parse_fallback_list(text: str, expected_n: int) -> List[str]:
    """
    Fallback extractor when the model doesn't return valid JSON.

    Parameters
    ----------
    text:
        Raw model output.
    expected_n:
        Maximum number of items to return.

    Returns
    -------
    items:
        Extracted lines with common list markers removed.

    Examples of supported formatting
    --------------------------------
    1) ...
    2. ...
    - ...
    * ...
    """
    t = strip_think_and_fences(text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    out: List[str] = []
    for ln in lines:
        ln2 = re.sub(r"^(\d+[\).\:-]\s+)", "", ln)
        ln2 = re.sub(r"^([-*•]\s+)", "", ln2)
        ln2 = ln2.strip()

        if not ln2:
            continue
        if ln2.lower().startswith(("here are", "sure", "json", "output", "examples:", "intent")):
            continue

        ln2 = ln2.strip().strip('"').strip("'").strip()
        if ln2:
            out.append(ln2)

    if len(out) > expected_n:
        out = out[:expected_n]
    return out


def extract_candidates(text: str, expected_n: int) -> List[str]:
    """
    Extract candidate utterances from the model output.

    Parameters
    ----------
    text:
        Raw model output.
    expected_n:
        The maximum number of candidates to return.

    Returns
    -------
    candidates:
        A list of strings extracted from the output.

    Raises
    ------
    ValueError:
        If neither JSON parsing nor fallback parsing yields any candidates.
    """
    arr = try_parse_json_array(text)
    if arr is not None and len(arr) > 0:
        return arr[:expected_n]

    fb = parse_fallback_list(text, expected_n)
    if fb:
        return fb

    raise ValueError("No JSON array found in output (and fallback parsing found nothing).")
