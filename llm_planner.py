from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from urllib import request, error


def load_dotenv(path: Optional[str] = None) -> bool:
    """Load values from a .env file into os.environ without overwriting existing values."""
    resolved = path or os.path.join(os.getcwd(), ".env")
    if not os.path.exists(resolved):
        return False
    loaded = False
    with open(resolved, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip(); value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value); loaded = True
    return loaded


from llm_model_experiment import MODEL_INTERFACE_CONTRACT, FEATURE_INTERFACE_CONTRACT

DEFAULT_SYSTEM_PROMPT = (
    "You are a disciplined ML research assistant for the KuaiRand-Pure benchmark: "
    "within-user ranking over logged impressions, binary label 'long_view'. "
    "The score to beat is the official FM baseline at validation primary 0.6016, "
    "where primary = mean(GAUC, nDCG@5). The theoretical ceiling is 0.8484 "
    "(27.1% of users are all-negative and score nDCG 0 for every model).\n\n"
    "Each turn choose ONE move, targeting whichever PIPELINE STAGE the history "
    "suggests is the bottleneck:\n"
    "  mode='tune'     -- re-run a registered experiment with different hyperparameters\n"
    "  mode='code'     -- write a new CandidateModel architecture (model stage)\n"
    "  mode='features' -- write a feature transform over the existing columns "
    "(feature stage); scored against a fixed reference architecture so the delta "
    "is attributable to the features alone\n"
    "  mode='train'    -- change the training recipe: loss "
    "('bce'|'weighted_bce' with pos_weight|'focal' with focal_gamma/focal_alpha), "
    "scheduler ('none'|'cosine'|'step'), lr, epochs, weight_decay, dropout, "
    "grad_clip, emb_dim, hidden, history_len, batch_size\n\n"
    "Do not keep proposing the same stage when it has stopped paying: the label is "
    "imbalanced and the dense block is only 10 columns, so feature and loss changes "
    "are often worth more than another architecture.\n\n"
    "Contract for mode='code':\n" + MODEL_INTERFACE_CONTRACT + "\n"
    "Contract for mode='features':\n" + FEATURE_INTERFACE_CONTRACT + "\n"
    "The history contains validation metrics and the errors of failed candidates; "
    "read them and do not repeat a rejected approach. Never reference hidden test "
    "data. Return only valid JSON with top-level 'proposal': "
    "{'mode':..., 'name':..., 'family':..., 'params':{...}, 'code':'...', "
    "'feature_code':'...', 'hypothesis':'...'}. "
    "Use 'name' only for tune, 'code' only for code, 'feature_code' only for features."
)


@dataclass
class LLMPlannerConfig:
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    deployment: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 4000
    force_code: bool = False
    max_retries: int = 3
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def disabled(cls) -> "LLMPlannerConfig":
        return cls(enabled=False)

    @classmethod
    def from_env(cls) -> "LLMPlannerConfig":
        api_key = os.getenv("KUAI_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        explicit_enabled = os.getenv("KUAI_LLM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        enabled = explicit_enabled or bool(api_key)
        force_code = os.getenv("KUAI_LLM_FORCE_CODE", "false").strip().lower() in {"1", "true", "yes", "on"}
        return cls(
            enabled=enabled,
            provider=os.getenv("KUAI_LLM_PROVIDER", "openai").strip() or "openai",
            model=os.getenv("KUAI_LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
            api_key=api_key,
            base_url=os.getenv("KUAI_LLM_BASE_URL") or None,
            api_version=os.getenv("KUAI_LLM_API_VERSION") or None,
            deployment=os.getenv("KUAI_LLM_DEPLOYMENT") or None,
            temperature=float(os.getenv("KUAI_LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("KUAI_LLM_MAX_TOKENS", "4000")),
            force_code=force_code,
            max_retries=max(1, int(os.getenv("KUAI_LLM_MAX_RETRIES", "3"))),
            system_prompt=os.getenv("KUAI_LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        )


@dataclass
class ProposedExperiment:
    name: str
    family: str
    params: Dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""
    code_diff: str = ""
    mode: str = "tune"
    code: str = ""
    feature_code: str = ""


class TruncatedResponseError(ValueError):
    """The provider stopped generating because the token cap was hit."""


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# torch / nn / F / np are pre-injected into the sandbox namespace, and the
# sandbox rejects the substring "import " outright. Small models write the
# habitual header anyway, so strip exactly those module-level imports (and
# nothing else -- a genuinely disallowed import must still be rejected).
_SAFE_IMPORT_RE = re.compile(r"^(?:import|from)\s+(?:torch|numpy)\b.*$")


def _extract_code(value: Any) -> str:
    """Accept generated code as a string, a list of source lines, or a fenced block.

    Small models routinely botch the escaping of a whole Python class inside a
    JSON string, so a JSON array of lines is an explicitly supported shape.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = "\n".join(str(part) for part in value)
    text = str(value)
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1)
    text = "\n".join(ln for ln in text.splitlines() if not _SAFE_IMPORT_RE.match(ln))
    return text.strip("\n")


_MODES = {"tune", "code", "features", "train"}
_DEFAULT_NAMES = {"code": "llm_generated", "features": "llm_features", "train": "llm_train"}


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict): return value
    if isinstance(value, (list, tuple)):
        out: Dict[str, Any] = {}
        for item in value:
            if isinstance(item, dict): out.update(item)
        return out
    return {}


def normalize_proposal(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, str):
        try: raw = json.loads(raw.strip())
        except json.JSONDecodeError as exc: raise ValueError(f"Proposal is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict): raise ValueError(f"Proposal must be a dictionary, got {type(raw).__name__}")
    proposal = raw.get("proposal", raw)
    if not isinstance(proposal, dict): raise ValueError("Proposal payload must decode to a dictionary")
    mode = str(proposal.get("mode") or "tune").strip().lower()
    code = _extract_code(proposal.get("code"))
    feature_code = _extract_code(proposal.get("feature_code") or proposal.get("features"))
    name = str(proposal.get("name") or proposal.get("experiment") or proposal.get("model") or "").strip()
    if mode not in _MODES:
        raise ValueError(f"unsupported proposal mode: {mode} (expected one of {sorted(_MODES)})")
    if mode == "tune" and not name:
        raise ValueError("mode='tune' requires the name of a registered experiment")
    if mode == "code" and not code.strip():
        raise ValueError("mode='code' requires non-empty code")
    if mode == "features" and not feature_code.strip():
        raise ValueError("mode='features' requires non-empty feature_code")
    if mode == "train" and not _coerce_mapping(proposal.get("params") or {}):
        raise ValueError("mode='train' requires a non-empty params object")
    default_family = {"code": "llm_code", "features": "llm_features",
                      "train": "llm_train"}.get(mode, "generic")
    return {
        "name": name or _DEFAULT_NAMES.get(mode, "llm_generated"),
        "family": str(proposal.get("family") or default_family),
        "params": _coerce_mapping(proposal.get("params") or proposal.get("arguments") or {}),
        "hypothesis": str(proposal.get("hypothesis") or proposal.get("reason") or ""),
        "code_diff": str(proposal.get("code_diff") or proposal.get("changes") or ""),
        "mode": mode,
        "code": code,
        "feature_code": feature_code,
    }


def resolve_experiment_spec(raw_proposal: Any, registry: Optional[Any] = None) -> ProposedExperiment:
    proposal = normalize_proposal(raw_proposal)
    name = proposal["name"]; family = proposal["family"]
    if registry is not None and proposal["mode"] != "code" and name in getattr(registry, "_items", {}):
        family = registry._items[name].family
    return ProposedExperiment(name, family, dict(proposal["params"]), proposal["hypothesis"],
                              proposal["code_diff"], proposal["mode"], proposal["code"],
                              proposal.get("feature_code", ""))


def _json_default(value: Any) -> Any:
    """Coerce values json.dumps cannot handle (numpy scalars, mainly).

    Validation metrics arrive as np.float32 straight off the evaluator, so
    every history summary containing a *successful* experiment used to raise
    TypeError -- a crash that only became reachable once a generated candidate
    actually trained.
    """
    for attr in ("item", "tolist"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    return str(value)


def _summarize_history(previous_records: Iterable[Dict[str, Any]]) -> str:
    rows = []
    for record in previous_records:
        metrics = record.get("metrics") or {}; valid = metrics.get("valid") or {}
        row = {"experiment": record.get("experiment"), "family": record.get("family"),
               "primary": valid.get("primary"), "gauc": valid.get("GAUC"),
               "ndcg": valid.get("nDCG@5"), "status": record.get("status")}
        if record.get("error"): row["error"] = str(record["error"]).strip().splitlines()[-1][:300]
        if record.get("family") == "llm_code" and record.get("code_diff"):
            row["previous_code_snippet"] = record["code_diff"][:400]
        rows.append(row)
    return json.dumps(rows[-10:], ensure_ascii=False, default=_json_default)


def summarize_for_prompt(previous_records: Iterable[Dict[str, Any]]) -> str:
    return "Recent validation history:\n" + _summarize_history(previous_records) + "\n\nPrioritize cheap, informative hypotheses."


class LLMPlanner:
    def __init__(self, config: Optional[LLMPlannerConfig] = None):
        self.config = config or LLMPlannerConfig.from_env()
        self.last_error: Optional[str] = None
        self.last_usage: Dict[str, Any] = {}
        self.last_proposal: Optional[Dict[str, Any]] = None
        self.last_raw_content: str = ""
        self.last_finish_reason: Optional[str] = None
        # Set by agent.py to record EVERY provider call, not just every
        # suggest_next_experiment() invocation.
        self.call_log_hook = None
        # Set by the controller to compile_candidate_model: validates generated
        # code (sandbox tokens + a dry-run forward pass) BEFORE the proposal is
        # returned, so a broken candidate is retried inside this call instead
        # of burning a whole controller iteration on a guaranteed failure.
        self.code_validator = None
        # Same idea for the feature stage: a transform that is not row-wise, or
        # that returns the wrong shape, is caught here rather than after a
        # multi-minute training run.
        self.feature_validator = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and (self.config.api_key or self.config.base_url))

    def _build_payload(self, registry: Any, previous_records: Iterable[Dict[str, Any]], force_code_attempt: bool = False, feedback: str = "") -> Dict[str, Any]:
        names = list(getattr(registry, "_items", {}).keys())
        task = (
            "You MUST choose mode='code' for this request. Do not return mode='tune'. "
            "Write a complete CandidateModel class satisfying the supplied interface contract. "
            "The class must be meaningfully different from the registered architectures."
            if force_code_attempt else
            "Choose ONE move: tune a registered experiment or write a full CandidateModel. Return the required JSON."
        )
        task += (
            " Do NOT write any import statements: torch, nn (torch.nn), "
            "F (torch.nn.functional) and np (numpy) are already in scope and the "
            "sandbox rejects the token 'import'. "
            " To avoid JSON escaping problems you MAY return 'code' as a JSON array of "
            "source lines instead of one string; it will be joined with newlines. "
            "Return exactly one JSON object and nothing else."
        )
        if feedback:
            task += (
                " YOUR PREVIOUS ATTEMPT FAILED: " + str(feedback)[:600] +
                " Fix that specific problem. If the response was truncated, write a "
                "shorter, more compact model."
            )
        return {"model": self.config.model, "temperature": self.config.temperature, "max_tokens": self.config.max_tokens,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": json.dumps({"allowed_experiments_for_mode_tune": names,
                        "history": summarize_for_prompt(previous_records), "task": task},
                        ensure_ascii=False, default=_json_default)}],
                **({"response_format": {"type": "json_object"}} if self.config.provider.lower() != "azure" else {})}

    def _build_request(self) -> tuple[str, Dict[str, str]]:
        provider = (self.config.provider or "openai").lower()
        base_url = self.config.base_url.rstrip("/") if self.config.base_url else ({"openai": "https://api.openai.com/v1", "openrouter": "https://openrouter.ai/api/v1"}.get(provider, "https://api.openai.com/v1"))
        if provider == "azure":
            if not self.config.api_key: raise ValueError("Azure provider requires KUAI_LLM_API_KEY")
            version = self.config.api_version or "2024-02-01"; deployment = self.config.deployment or self.config.model
            return f"{base_url}/openai/deployments/{deployment}/chat/completions?api-version={version}", {"Content-Type":"application/json","api-key":self.config.api_key}
        url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        return url, {"Content-Type":"application/json","Authorization":f"Bearer {self.config.api_key}"}

    def _log_call(self, record: Dict[str, Any]) -> None:
        """Emit one telemetry record per provider HTTP call (best effort)."""
        record["duration_sec"] = time.time() - record.get("timestamp", time.time())
        hook = self.call_log_hook
        if hook is None:
            return
        try:
            hook(dict(record))
        except Exception:
            pass

    def _do_call(self, payload: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
        url, headers = self._build_request()
        req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM provider request failed: {details}") from exc

        self.last_usage = dict(data.get("usage") or {})
        choice = (data.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason") or choice.get("native_finish_reason")
        self.last_finish_reason = finish_reason
        message = choice.get("message", {})
        content = message.get("content") or data.get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        self.last_raw_content = content
        record["usage"] = dict(self.last_usage)
        record["finish_reason"] = finish_reason
        record["response_chars"] = len(content)

        # A truncated response is the single most common cause of "invalid
        # JSON" from a code-writing planner. Name it explicitly instead of
        # letting json.JSONDecodeError take the blame.
        if finish_reason == "length":
            raise TruncatedResponseError(
                "LLM response hit the token cap (finish_reason='length'): "
                f"max_tokens={payload.get('max_tokens')}, "
                f"completion_tokens={self.last_usage.get('completion_tokens')}. "
                "Raise KUAI_LLM_MAX_TOKENS or ask for a more compact model."
            )
        if not content.strip():
            raise ValueError(f"LLM returned an empty response (finish_reason={finish_reason!r})")

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            start, end = content.find("{"), content.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end + 1])
                except json.JSONDecodeError as inner:
                    raise ValueError(
                        f"LLM response was not valid JSON ({inner}); "
                        f"finish_reason={finish_reason!r}; see raw_content in llm_calls.jsonl"
                    ) from inner
            raise ValueError(
                f"LLM response was not JSON ({exc}); finish_reason={finish_reason!r}; "
                f"first 400 chars: {content[:400]!r}"
            ) from exc

    def _call_provider(self, payload: Dict[str, Any], attempt: int = 0) -> Dict[str, Any]:
        if not self.enabled or not self.config.api_key:
            raise RuntimeError("LLM planner is disabled or missing API configuration")
        self.last_raw_content = ""
        self.last_finish_reason = None
        record: Dict[str, Any] = {
            "timestamp": time.time(),
            "attempt": attempt,
            "provider": self.config.provider,
            "model": self.config.model,
            "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
            "force_code": self.config.force_code,
        }
        try:
            parsed = self._do_call(payload, record)
        except Exception as exc:
            record["success"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            # The raw text is the whole point of this log: without it a parse
            # failure is unfalsifiable guesswork.
            record["raw_content"] = (self.last_raw_content or "")[:8000]
            self._log_call(record)
            raise
        record["success"] = True
        self._log_call(record)
        return parsed

    def suggest_next_experiment(self, registry: Any, previous_records: Iterable[Dict[str, Any]]) -> Optional[ProposedExperiment]:
        self.last_usage = {}
        self.last_proposal = None
        self.last_error = None
        self.last_raw_content = ""
        self.last_finish_reason = None
        if not self.enabled:
            return None

        previous = list(previous_records)
        # Retries apply whether or not force_code is on: a candidate rejected by
        # the dry-run validator is worth one more turn in every mode.
        attempts = max(1, int(self.config.max_retries))
        feedback = ""
        for attempt in range(attempts):
            payload = self._build_payload(
                registry, previous,
                force_code_attempt=self.config.force_code,
                feedback=feedback,
            )
            # Retrying an identical prompt at the same temperature reproduces
            # the same failure; nudge it and hand back the error text.
            if attempt:
                payload["temperature"] = min(1.0, float(self.config.temperature) + 0.2 * attempt)
            try:
                response = self._call_provider(payload, attempt=attempt)
                normalized = normalize_proposal(response)
                self.last_proposal = normalized
                if self.config.force_code and normalized["mode"] != "code":
                    self.last_error = "LLM returned mode='tune' while KUAI_LLM_FORCE_CODE=true"
                    feedback = self.last_error
                    continue
                if normalized["mode"] == "code" and self.code_validator is not None:
                    try:
                        self.code_validator(normalized["code"])
                    except Exception as exc:
                        self.last_error = f"generated code rejected: {type(exc).__name__}: {exc}"
                        feedback = self.last_error
                        continue
                if normalized["mode"] == "features" and self.feature_validator is not None:
                    try:
                        self.feature_validator(normalized["feature_code"])
                    except Exception as exc:
                        self.last_error = f"feature transform rejected: {type(exc).__name__}: {exc}"
                        feedback = self.last_error
                        continue
                self.last_error = None
                return resolve_experiment_spec(normalized, registry)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                feedback = self.last_error

        return None
