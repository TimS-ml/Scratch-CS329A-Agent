"""Lightweight disk cache + call log for all LLM requests.

Goals
-----
1. Persist every model request/response to ``.cache/llm/calls.jsonl`` so the raw
   outputs can be analysed later.
2. Act as a read-through cache so re-running notebooks (debug pass, full pass,
   doc regeneration) does not re-pay for identical calls.

Diversity-safe keying
----------------------
Temperature>0 sampling intentionally wants *different* completions for the same
prompt. Callers therefore pass an ``occ`` (occurrence index) that distinguishes
the N identical prompts inside a single sampling batch. Each of the N samples is
cached under its own key, so reruns reproduce the same N *diverse* samples.

Configuration (env vars)
-------------------------
* ``HW_CACHE``      "0" disables caching entirely (still no logging). Default on.
* ``HW_CACHE_DIR``  override cache root. Default ``<cwd>/.cache/llm``.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import threading
import time
from typing import Any, Callable, Optional

_LOCK = threading.Lock()


def _enabled() -> bool:
    return os.getenv("HW_CACHE", "1").lower() not in ("0", "false", "no", "")


def _cache_root() -> pathlib.Path:
    root = os.getenv("HW_CACHE_DIR")
    p = pathlib.Path(root) if root else (pathlib.Path.cwd() / ".cache" / "llm")
    (p / "responses").mkdir(parents=True, exist_ok=True)
    return p


def make_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _read(key: str) -> Optional[dict]:
    fp = _cache_root() / "responses" / f"{key}.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text())
        except Exception:
            return None
    return None


def _write(key: str, record: dict) -> None:
    root = _cache_root()
    fp = root / "responses" / f"{key}.json"
    with _LOCK:
        fp.write_text(json.dumps(record, ensure_ascii=False, default=str))
        with open(root / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def cached(payload: dict, producer: Callable[[], Any], tag: str = "") -> Any:
    """Return cached response for ``payload`` else run ``producer`` and store it.

    Note: we do NOT cache ``None`` responses (treated as transient API failure)
    so the next run gets a real retry instead of replaying the failure.
    """
    if not _enabled():
        return producer()
    key = make_key(payload)
    hit = _read(key)
    if hit is not None and "response" in hit:
        return hit["response"]
    t0 = time.time()
    resp = producer()
    if resp is None:
        return None  # don't poison the cache with transient failures
    record = {
        "key": key,
        "tag": tag,
        "request": payload,
        "response": resp,
        "ts": time.time(),
        "latency_s": round(time.time() - t0, 3),
    }
    _write(key, record)
    return resp
