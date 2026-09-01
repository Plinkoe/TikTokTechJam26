from __future__ import annotations

import json
import os
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


from llm_model_experiment import MODEL_INTERFACE_CONTRACT

DEFAULT_SYSTEM_PROMPT = (
    "You are a disciplined ML research assistant for the KuaiRand benchmark. "
    "Each turn choose ONE move: mode='tune' to tune an existing registered experiment, "
    "or mode='code' to write a brand-new CandidateModel architecture.\n\n"
    "The interface contract for mode='code' is:\n" + MODEL_INTERFACE_CONTRACT + "\n"
    "The history contains validation metrics and errors from failed candidates. "
    "Ignore hidden test data. Return only valid JSON with top-level 'proposal': "
    "{'mode':'tune'|'code','name':...,'family':...,'params':{...},'code':'...','hypothesis':'...'}. "
    "Omit code for tune; omit name for code."
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
    max_tokens: int = 1200
    force_code: bool = False
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
            max_tokens=int(os.getenv("KUAI_LLM_MAX_TOKENS", "1200")),
            force_code=force_code,
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
    code = str(proposal.get("code") or "")
    name = str(proposal.get("name") or proposal.get("experiment") or proposal.get("model") or "").strip()
    if mode not in {"tune", "code"}: raise ValueError(f"unsupported proposal mode: {mode}")
    if mode != "code" and not name: raise ValueError("Proposal missing a valid name")
    if mode == "code" and not code.strip(): raise ValueError("mode='code' requires non-empty code")
    return {
        "name": name or "llm_generated",
        "family": str(proposal.get("family") or ("llm_code" if mode == "code" else "generic")),
        "params": _coerce_mapping(proposal.get("params") or proposal.get("arguments") or {}),
        "hypothesis": str(proposal.get("hypothesis") or proposal.get("reason") or ""),
        "code_diff": str(proposal.get("code_diff") or proposal.get("changes") or ""),
        "mode": mode,
        "code": code,
    }


def resolve_experiment_spec(raw_proposal: Any, registry: Optional[Any] = None) -> ProposedExperiment:
    proposal = normalize_proposal(raw_proposal)
    name = proposal["name"]; family = proposal["family"]
    if registry is not None and proposal["mode"] != "code" and name in getattr(registry, "_items", {}):
        family = registry._items[name].family
    return ProposedExperiment(name, family, dict(proposal["params"]), proposal["hypothesis"], proposal["code_diff"], proposal["mode"], proposal["code"])


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
    return json.dumps(rows[-10:], ensure_ascii=False)


def summarize_for_prompt(previous_records: Iterable[Dict[str, Any]]) -> str:
    return "Recent validation history:\n" + _summarize_history(previous_records) + "\n\nPrioritize cheap, informative hypotheses."


class LLMPlanner:
    def __init__(self, config: Optional[LLMPlannerConfig] = None):
        self.config = config or LLMPlannerConfig.from_env()
        self.last_error: Optional[str] = None
        self.last_usage: Dict[str, Any] = {}
        self.last_proposal: Optional[Dict[str, Any]] = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and (self.config.api_key or self.config.base_url))

    def _build_payload(self, registry: Any, previous_records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        names = list(getattr(registry, "_items", {}).keys())
        if self.config.force_code:
            task = (
                "For this run, mode='code' is REQUIRED. Do not tune a registered experiment. "
                "Write a complete, concise CandidateModel class satisfying the supplied interface contract. "
                "Use the existing causal-history inputs if useful, but make a genuinely different architecture "
                "from the registered HistoryDeepFM model. Return the class source in 'code' and a concise hypothesis."
            )
        else:
            task = "Choose ONE move: tune a registered experiment or write a full CandidateModel. Return the required JSON."
        return {"model": self.config.model, "temperature": self.config.temperature, "max_tokens": self.config.max_tokens,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": json.dumps({"allowed_experiments_for_mode_tune": names,
                        "history": summarize_for_prompt(previous_records), "task": task}, ensure_ascii=False)}],
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

    def _call_provider(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or not self.config.api_key: raise RuntimeError("LLM planner is disabled or missing API configuration")
        url, headers = self._build_request()
        req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM provider request failed: {details}") from exc
        self.last_usage = dict(data.get("usage") or {})
        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content") or data.get("content") or ""
        if isinstance(content, list): content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        try: return json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if start >= 0 and end > start: return json.loads(content[start:end + 1])
            raise ValueError(f"LLM response was not JSON: {content[:400]}")

    def suggest_next_experiment(self, registry: Any, previous_records: Iterable[Dict[str, Any]]) -> Optional[ProposedExperiment]:
        self.last_usage = {}
        self.last_proposal = None
        if not self.enabled: return None
        try:
            response = self._call_provider(self._build_payload(registry, previous_records))
            self.last_proposal = normalize_proposal(response)
            spec = resolve_experiment_spec(self.last_proposal, registry)
            self.last_error = None
            return spec
        except Exception as exc:
            self.last_error = str(exc)
            return None
