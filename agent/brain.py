"""Lane B, task B2: brain client. Muse (Meta Model API, OpenAI-compatible) writes the code.

    python -m agent.brain --ping      # toy turn, prints the parsed JSON, logs to runs/ping/brain_calls.jsonl
    python -m agent.brain --models    # GET /v1/models, prints the newest Muse Spark id
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from agent.calllog import log_call

load_dotenv()

BACKOFF_S = (1.0, 2.0, 4.0)          # retry 429 / 5xx after 1 s, 2 s, 4 s
MAX_TOKENS = 16000                   # Muse Spark is a reasoning model: its thinking counts against this

PROVIDERS = {
    "muse": {"base_url": "META_BASE_URL", "api_key": "MODEL_API_KEY", "model": "META_MODEL",
             "default_model": "muse-spark-1.1"},
    # Optional, off the critical path: only if the SubQ booth enables chat for our key.
    "subq_chat": {"base_url": "SUBQ_BASE_URL", "api_key": "SUBQ_API_KEY", "model": "SUBQ_CHAT_MODEL",
                  "default_model": "subq-preview"},
}

# Per-provider memory of unsupported parameters, shared across Brain instances for the session.
_UNSUPPORTED: dict[str, set[str]] = {}


class BrainError(RuntimeError):
    pass


def extract_json(text: str) -> dict:
    """Raw JSON object, or a ```json fence, or the first balanced {...}."""
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.S):
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
                    break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in the reply")


def _retryable(exc: Exception) -> bool:
    import openai

    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def _rejects_param(exc: Exception, name: str) -> bool:
    import openai

    return (isinstance(exc, openai.BadRequestError) or getattr(exc, "status_code", None) in (400, 422)) and \
        name in str(exc).lower()


class Brain:
    def __init__(self, provider: str = "muse", *, run_id: str = "adhoc", store=None, live: bool = False,
                 model: str | None = None) -> None:
        if provider not in PROVIDERS:
            raise BrainError(f"unknown brain {provider!r}; expected one of {sorted(PROVIDERS)}")
        import openai

        cfg = PROVIDERS[provider]
        key = os.getenv(cfg["api_key"])
        if not key:
            raise BrainError(f"{cfg['api_key']} is not set in .env")
        self.provider = provider
        self.model = model or os.getenv(cfg["model"]) or cfg["default_model"]
        self.run_id = run_id
        self.store = store
        self.timeout = 35.0 if live else 60.0
        self.client = openai.OpenAI(base_url=os.getenv(cfg["base_url"]), api_key=key,
                                    timeout=self.timeout, max_retries=0)
        self.last_finish_reason: str | None = None
        _UNSUPPORTED.setdefault(provider, set())

    # ------------------------------------------------------------------ raw call
    def complete(self, messages: list[dict], *, json_mode: bool = True, temperature: float = 0.2) -> str:
        """One chat completion with 429/5xx backoff and unsupported-param fallback. Logs every attempt.
        Sets self.last_finish_reason ("length" = the reply was cut off by max_tokens)."""
        unsupported = _UNSUPPORTED[self.provider]
        attempt = 0
        while True:
            kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": MAX_TOKENS}
            if "temperature" not in unsupported:
                kwargs["temperature"] = temperature
            if json_mode and "response_format" not in unsupported:
                kwargs["response_format"] = {"type": "json_object"}
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                latency = int((time.time() - t0) * 1000)
                self._log(latency, None, None, ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
                for p in ("response_format", "temperature"):
                    if p in kwargs and _rejects_param(e, p):
                        unsupported.add(p)          # remember for the session, retry once without it
                        break
                else:
                    if _retryable(e) and attempt < len(BACKOFF_S):
                        time.sleep(BACKOFF_S[attempt])
                        attempt += 1
                        continue
                    raise BrainError(f"{self.provider} call failed: {type(e).__name__}: {e}") from e
                continue
            latency = int((time.time() - t0) * 1000)
            u = getattr(r, "usage", None)
            self.last_finish_reason = getattr(r.choices[0], "finish_reason", None)
            content = r.choices[0].message.content or ""
            self._log(latency, getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None), ok=True,
                      finish_reason=self.last_finish_reason, content_chars=len(content))
            return content

    def _log(self, latency_ms, prompt_tokens, completion_tokens, ok: bool, error: str | None = None,
             **extra) -> None:
        log_call(self.run_id, {"provider": self.provider, "model": self.model, "latency_ms": latency_ms,
                               "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                               "ok": ok, "error": error, **extra}, self.store)

    # ------------------------------------------------------------------ structured call
    def complete_json(self, messages: list[dict], model_cls: type[BaseModel], *, normalize=None,
                      max_fix_retries: int = 2) -> dict:
        """Call, extract JSON, validate with `model_cls`; on failure retry ≤ 2× with the error appended."""
        msgs = list(messages)
        last_err = ""
        for _ in range(max_fix_retries + 1):
            text = self.complete(msgs)
            try:
                obj = extract_json(text)
                if normalize:
                    obj = normalize(obj)
                return model_cls.model_validate(obj).model_dump()
            except (ValueError, ValidationError) as e:
                last_err = str(e)[:1500]
                if self.last_finish_reason == "length":
                    # Cut off mid-thought: resending the partial text would only make it longer.
                    last_err = "reply cut off by the token limit before the JSON was finished"
                    msgs = list(messages) + [{"role": "user", "content":
                                              "Your previous reply ran out of tokens before the JSON. Think briefly "
                                              "(a few sentences at most), then output the JSON object immediately."}]
                    continue
                msgs = msgs + [
                    {"role": "assistant", "content": text[:6000]},
                    {"role": "user", "content": "Your reply was not valid. Error:\n" + last_err +
                     "\nRespond again with ONLY one JSON object matching the schema."},
                ]
        raise BrainError(f"invalid JSON after {max_fix_retries + 1} attempts: {last_err[:300]}")

    def turn(self, messages: list[dict], action_names: list[str] | None = None) -> dict:
        """One agent turn -> validated protocol.BrainTurn as a dict."""
        from agent.protocol import BrainTurn, normalize_turn

        return self.complete_json(messages, BrainTurn, normalize=lambda o: normalize_turn(o, action_names))

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list().data)


def newest_muse_spark(ids: list[str]) -> str | None:
    """Highest muse-spark-X.Y version, ignoring variants such as '-contributor'."""
    best = None
    for i in ids:
        m = re.fullmatch(r"muse-spark-(\d+)\.(\d+)", i)
        if m and (best is None or (int(m[1]), int(m[2])) > best[0]):
            best = ((int(m[1]), int(m[2])), i)
    return best[1] if best else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true")
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--brain", default=os.getenv("BRAIN", "muse"))
    args = ap.parse_args()
    brain = Brain(args.brain, run_id="ping")
    if args.models:
        ids = brain.list_models()
        print("models:", ids)
        print("newest Muse Spark:", newest_muse_spark(ids), "| configured META_MODEL:", brain.model)
    if args.ping or not args.models:
        from agent.prompts import system_prompt

        msgs = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": "This is a connectivity test. Return a valid turn with summary "
                 "'ping ok', one inferred ledger rule 'test rule', no program, no probes, done=true."}]
        t0 = time.time()
        out = brain.turn(msgs)
        print(json.dumps(out, indent=2))
        print(f"ok in {time.time() - t0:.1f}s; logged to runs/ping/brain_calls.jsonl")


if __name__ == "__main__":
    main()
