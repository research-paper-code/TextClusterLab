"""
LLM-related functions for CLINC augmentation.

This module owns:
- Prompt/message construction
- (Optional) guided decoding schema
- Generation configuration (GenConfig)
- The core generation loop that calls an OpenAI-compatible vLLM endpoint
"""

from __future__ import annotations

import random
import time
import uuid
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .utils import clean_generated_text, extract_candidates, normalize, is_bad_candidate

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Qwen setting

def build_messages(
    rng: random.Random,
    intent_label: str,
    class_texts: List[str],
    shots_k: int,
    n_to_generate: int,
    avoid_texts: List[str],
) -> List[Dict[str, str]]:
    """
    Construct Chat Completions messages for intent-class utterance augmentation.

    Parameters
    ----------
    rng:
        Seeded random generator used for selecting few-shot examples.
    intent_label:
        The class/intent name used as conditioning.
    class_texts:
        Real utterances from the dataset belonging to this intent.
    shots_k:
        Number of few-shot examples to include in the prompt.
    n_to_generate:
        Number of new utterances requested from the model in this call.
    avoid_texts:
        Recently generated utterances to discourage exact repetition.

    Returns
    -------
    messages:
        List of {"role": "...", "content": "..."} dicts compatible with OpenAI chat API.

    Notes
    -----
    The system prompt enforces "JSON array only" to make parsing easier.
    Print at line 260.
    """
    shots = rng.sample(class_texts, min(shots_k, len(class_texts)))
    nonce = str(uuid.uuid4())

    system = (
        "You generate synthetic intent-classification utterances.\n"
        "Return ONLY the final answer.\n"
        "Do NOT include analysis, thinking, or explanations.\n"
        "Output must be a JSON array of strings (no markdown).\n"
        f"Return exactly {n_to_generate} strings.\n"
        "- Do NOT copy any provided example verbatim.\n"
        "- Keep same intent; vary wording/entities/length.\n"
        "- Use normal spaces between words (no underscores '_' and no '▁' markers).\n"
        "- Each string must be a natural-language utterance (not character lists or repeated letters).\n"
        "- Do NOT output strings that are mostly single-letter tokens or repeated characters.\n"
        "- Each string should have at least 4 words.\n"
        "- Do NOT output timestamps, IDs, counters, alphabet ranges, or percentage lists.\n"
        "- Avoid bracketed character sequences like [A] [B] [C] or long symbol runs.\n"
    )

    avoid_block = ""
    if avoid_texts:
        avoid_block = (
            "\nAvoid repeating any of these exactly:\n"
            + "\n".join([f"- {x}" for x in avoid_texts])
        )

    user = (
        f"Nonce: {nonce}\n"
        f"Intent label: {intent_label}\n\n"
        "Examples (same intent):\n"
        + "\n".join([f"- {s}" for s in shots])
        + avoid_block
        + "\n\nGenerate now as a JSON array of strings."
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Llama setting

# def build_messages(
#     rng: random.Random,
#     intent_label: str,
#     class_texts: List[str],
#     shots_k: int,
#     n_to_generate: int,
#     avoid_texts: List[str],
# ) -> List[Dict[str, str]]:
#     """
#     Construct Chat Completions messages for intent-class utterance augmentation.
#     """
#     # Keep shots small & diverse; too many examples increases drift.
#     shots = rng.sample(class_texts, min(shots_k, len(class_texts)))

#     system = (
#         "You generate synthetic intent-classification utterances.\n"
#         "Return ONLY the final answer.\n"
#         "Output must be a JSON array of strings (no markdown).\n"
#         f"Return exactly {n_to_generate} strings.\n"
#         "\n"
#         "HARD RULES for EACH string:\n"
#         "- One utterance only (single line). No newline characters.\n"
#         "- 6 to 20 words.\n"
#         "- Max 140 characters.\n"
#         "- Natural conversational English.\n"
#         "- Do NOT repeat any phrase of 4+ words.\n"
#         "- Do NOT include markup/tags/code (no <...>, no JSON objects).\n"
#         "- Do NOT include weird character runs (e.g., 'vvvvv', 'a b c').\n"
#         "- Do NOT copy any provided example verbatim.\n"
#         "- Use normal spaces between words (no underscores '_' and no '▁' markers).\n"
#         "\n"
#         "Return ONLY the JSON array. No extra text."
#     )

#     avoid_block = ""
#     if avoid_texts:
#         # Keep avoid list short to reduce prompt size; last-k is handled by caller.
#         avoid_block = (
#             "\nAvoid repeating any of these exactly:\n"
#             + "\n".join([f"- {x}" for x in avoid_texts])
#         )

#     user = (
#         f"Intent label: {intent_label}\n\n"
#         "Examples (same intent, for style only):\n"
#         + "\n".join([f"- {s}" for s in shots])
#         + avoid_block
#         + "\n\n"
#         f"Now generate {n_to_generate} NEW utterances as a JSON array of strings."
#     )

#     return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Qwen setting
def guided_json_schema(expected_n: int) -> Dict[str, Any]:
    """
    Create a JSON schema used by vLLM structured output constraints (if supported).

    Parameters
    ----------
    expected_n:
        Required number of items in the JSON array.

    Returns
    -------
    schema:
        A JSON-schema-like dict for a fixed-length array of strings.

    Notes
    -----
    Many vLLM builds accept `extra_body={"structured_outputs": {"json": schema}}` and will force
    strictly valid JSON output. If your server rejects it, we retry without it.
    """
    return {
        "type": "array",
        "minItems": expected_n,
        "maxItems": expected_n,
        "items": {"type": "string"},
    }
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Llama setting
# def guided_json_schema(expected_n: int) -> Dict[str, Any]:
#     return {
#         "type": "array",
#         "minItems": expected_n,
#         "maxItems": expected_n,
#         "items": {
#             "type": "string",
#             "minLength": 10,
#             "maxLength": 160,
#             # single-line and disallow <...> tags
#             "pattern": r"^[^\n\r<>]+$",
#         },
#     }
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@dataclass(frozen=True)
class GenConfig:
    """
    Container for all generation-related parameters.

    This keeps function signatures clean while still letting you pass many knobs.

    Fields
    ------
    n_new:
        Number of synthetic examples to generate per class.
    batch:
        Candidates requested per API call.
    n_shots:
        Few-shot examples shown in the prompt.
    avoid_last_k:
        Include last K generated items in the prompt as an "avoid" list.
    max_attempts:
        Max number of API calls per class.

    temperature_start, top_p_start, presence_penalty_start, frequency_penalty_start:
        Initial decoding parameters. They are dynamically adjusted upward if
        the model keeps producing duplicates or parsing fails.
    max_tokens:
        Max tokens per response.
    seed:
        Random seed for shot selection and random label selection (when applicable).
    """
    n_new: int
    batch: int
    n_shots: int
    avoid_last_k: int
    max_attempts: int

    temperature_start: float
    top_p_start: float
    presence_penalty_start: float
    frequency_penalty_start: float
    max_tokens: int

    seed: int


def pick_model_id(client: Any, requested_model: Optional[str]) -> str:
    """
    Choose which model id to use for generation.

    Parameters
    ----------
    client:
        OpenAI client created with base_url pointing to vLLM's /v1 endpoint.
    requested_model:
        If provided, use it directly; otherwise auto-select the first model
        returned from /v1/models.

    Returns
    -------
    model_id:
        The model id string to pass into chat.completions.create.

    Raises
    ------
    RuntimeError:
        If auto-selection is requested but /v1/models returns no models.
    """
    if requested_model:
        return requested_model

    models = client.models.list()
    if not getattr(models, "data", None):
        raise RuntimeError("No models returned from /v1/models. Check your vLLM server.")
    return models.data[0].id


def generate_for_label(
    client: Any,
    model_id: str,
    label: str,
    class_texts: List[str],
    cfg: GenConfig,
) -> List[str]:
    """
    Generate synthetic utterances for one label/class.

    Parameters
    ----------
    client:
        OpenAI client pointing to your vLLM endpoint.
    model_id:
        Model name/id to use.
    label:
        The intent label being augmented.
    class_texts:
        Real utterances from this label (used for few-shot examples and dedupe).
    cfg:
        Generation configuration.

    Returns
    -------
    generated:
        List of accepted synthetic utterances (length up to cfg.n_new).

    Behavior
    --------
    - Repeatedly requests cfg.batch candidates until cfg.n_new accepted
      or cfg.max_attempts reached.
    - Dedupes against (a) original class_texts and (b) previously generated.
    - Cleans tokenization artifacts (underscores, SentencePiece markers, etc.)
    - If structured_outputs constraints are rejected by the server, retries without them.
    - If stuck, increases temperature / penalties to encourage diversity.
    """
    rng = random.Random(cfg.seed)

    existing_set = {normalize(t) for t in class_texts}
    generated: List[str] = []
    generated_set: set[str] = set()

    temperature = cfg.temperature_start
    top_p = cfg.top_p_start
    presence_penalty = cfg.presence_penalty_start
    frequency_penalty = cfg.frequency_penalty_start

    attempts = 0
    while len(generated) < cfg.n_new and attempts < cfg.max_attempts:
        attempts += 1
        need = min(cfg.batch, cfg.n_new - len(generated))
        avoid_texts = generated[-cfg.avoid_last_k:] if cfg.avoid_last_k > 0 else []

        messages = build_messages(
            rng=rng,
            intent_label=label,
            class_texts=class_texts,
            shots_k=cfg.n_shots,
            n_to_generate=need,
            avoid_texts=avoid_texts,
        )

        # Print the exact prompt/messages sent to the LLM for transparency/debugging.
        # Using JSON formatting keeps multi-role messages readable in logs.


        # print(
        #     "[PROMPT]",
        #     f"label={label}",
        #     f"attempt={attempts}",
        #     "messages=\n" + json.dumps(messages, indent=2),
        # )

        extra_body = {"structured_outputs": {"json": guided_json_schema(need)}}

        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                max_tokens=cfg.max_tokens,
                extra_body=extra_body,
            )
        except Exception as e:
            # Older vLLM builds or non-vLLM backends may reject structured_outputs.
            # In that case, retry without constraints and rely on robust parsing.
            print(f"[WARN] structured_outputs not accepted, retrying without it. ({e})")
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                max_tokens=cfg.max_tokens,
            )

        out_text = resp.choices[0].message.content or ""

        try:
            candidates = extract_candidates(out_text, expected_n=need)
        except Exception as e:
            print(f"[WARN] label={label} parse failed (attempt {attempts}). Error: {e}")
            temperature = min(1.3, temperature + 0.15)
            top_p = max(0.85, top_p - 0.03)
            presence_penalty = min(1.0, presence_penalty + 0.1)
            frequency_penalty = min(1.0, frequency_penalty + 0.1)
            time.sleep(0.15)
            continue

        accepted = 0
        for c in candidates:
            c = clean_generated_text(c)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            # llama setting: append this part
            # HARD quality gates (prevents long essays from being accepted)
            # words = c.split()
            # if not (6 <= len(words) <= 20):
            #     continue
            # if len(c) > 140:
            #     continue
            # if "\n" in c or "\r" in c:
            #     continue
            # if "<" in c or ">" in c:
            #     continue
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if not c:
                continue
            if is_bad_candidate(c):
                continue

            key = normalize(c)
            if not key:
                continue
            if key in existing_set:
                continue
            if key in generated_set:
                continue

            generated.append(c)
            generated_set.add(key)
            accepted += 1
            if len(generated) >= cfg.n_new:
                break

        print(
            f"[INFO] label={label} attempt={attempts} candidates={len(candidates)} "
            f"accepted={accepted} total={len(generated)}/{cfg.n_new} "
            f"(temp={temperature:.2f}, top_p={top_p:.2f})"
        )

        if accepted == 0:
            temperature = min(1.3, temperature + 0.15)
            top_p = max(0.85, top_p - 0.03)
            presence_penalty = min(1.0, presence_penalty + 0.1)
            frequency_penalty = min(1.0, frequency_penalty + 0.1)

    if len(generated) < cfg.n_new:
        print(f"[WARN] label={label} only generated {len(generated)}/{cfg.n_new} after {attempts} attempts.")

    return generated


# =========================
# High-level API (no CLI)
# =========================

from typing import Union, Tuple
import os

from openai import OpenAI

from .dataprocesser import group_texts_by_label, infer_text_and_label_keys


def _unpack_rows(data: Union[dict, list]) -> Tuple[list, dict]:
    """
    Normalize JSON-style input `data` into (rows, cfg).

    Parameters
    ----------
    data:
        - list[dict] rows, OR
        - dict with {"rows": list[dict], ...optional config...}

    Returns
    -------
    (rows, cfg):
        rows: list[dict]
        cfg:  dict (may be empty) with optional generation/server config
    """
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        rows = data["rows"]
        cfg = {k: v for k, v in data.items() if k != "rows"}
        return rows, cfg
    raise ValueError("data must be list[dict] or dict with key 'rows' as list[dict].")


def generate_synthetic(
    data: Union[dict, list],
    n_new: int,
    model_select: str | None,
) -> list[dict]:
    """
    Main API: generate synthetic utterances for a **single-class** input dataset.

    Inputs
    ------
    data:
        JSON-style input containing exactly ONE label/class.
        Supported forms:
          - list[dict] rows
          - {"rows": list[dict], ...optional config...}

        Optional config keys (only if dict form is used):
          - base_url: str (default: env VLLM_BASE_URL or http://127.0.0.1:6006/v1)
          - api_key:  str (default: env VLLM_API_KEY or "EMPTY")
          - batch: int (default 20)
          - n_shots: int (default 20)
          - avoid_last_k: int (default 30)
          - max_attempts: int (default 60)
          - temperature_start: float (default 0.9)
          - top_p_start: float (default 0.95)
          - presence_penalty_start: float (default 0.6)
          - frequency_penalty_start: float (default 0.6)
          - max_tokens: int (default 1200)
          - seed: int (default 42)

    n_new:
        Number of synthetic examples to generate (for the single class).
    model_select:
        LLM model id/name to use (e.g., "Qwen3-8B"). If None, auto-selects
        the first model returned by GET /v1/models.

    Returns
    -------
    records:
        list[dict] generated records. Each record is {text_key: "...", label_key: "..."}.

    Raises
    ------
    ValueError:
        If input rows contain 0 or >1 labels/classes.
    """
    rows, cfg = _unpack_rows(data)

    text_key, label_key = infer_text_and_label_keys(rows)
    label2texts = group_texts_by_label(rows, text_key, label_key)

    labels = sorted(label2texts.keys())
    if len(labels) != 1:
        raise ValueError(f"Input data must contain exactly one class/label, found {len(labels)}: {labels}")
    label = labels[0]
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Qwen configure
    gen_cfg = GenConfig(
        n_new=int(n_new),
        batch=int(cfg.get("batch", 20)),
        n_shots=int(cfg.get("n_shots", 10000)),
        avoid_last_k=int(cfg.get("avoid_last_k", 3)),
        max_attempts=int(cfg.get("max_attempts", 60)),
        temperature_start=float(cfg.get("temperature_start", os.getenv("VLLM_TEMPERATURE", 0.95))),
        top_p_start=float(cfg.get("top_p_start", os.getenv("VLLM_TOP_P", 0.8))),
        presence_penalty_start=float(cfg.get("presence_penalty_start", 0.6)),
        frequency_penalty_start=float(cfg.get("frequency_penalty_start", 0.6)),
        max_tokens=int(cfg.get("max_tokens", 1000)),
        seed=int(cfg.get("seed", 42)),
    )
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Llama configure
    # gen_cfg = GenConfig(
    #     n_new=int(n_new),
    #     batch=int(cfg.get("batch", 10)),
    #     n_shots=int(cfg.get("n_shots", 8)),          # was 10000
    #     avoid_last_k=int(cfg.get("avoid_last_k", 30)),# was 5
    #     max_attempts=int(cfg.get("max_attempts", 60)),

    #     temperature_start=float(cfg.get("temperature_start", 0.9)),
    #     top_p_start=float(cfg.get("top_p_start", 0.95)),
    #     presence_penalty_start=float(cfg.get("presence_penalty_start", 0.25)),
    #     frequency_penalty_start=float(cfg.get("frequency_penalty_start", 0.35)),

    #     max_tokens=int(cfg.get("max_tokens", 220)),  # was 1200
    #     seed=int(cfg.get("seed", 42)),
    # )
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    base_url = str(cfg.get("base_url") or os.getenv("VLLM_BASE_URL") or "http://127.0.0.1:6006/v1")
    api_key = str(cfg.get("api_key") or os.getenv("VLLM_API_KEY") or "EMPTY")

    client = OpenAI(base_url=base_url, api_key=api_key)
    model_id = pick_model_id(client, model_select)

    class_texts = label2texts[label]
    generated = generate_for_label(
        client=client,
        model_id=model_id,
        label=label,
        class_texts=class_texts,
        cfg=gen_cfg,
    )

    return [{text_key: t, label_key: label} for t in generated]
