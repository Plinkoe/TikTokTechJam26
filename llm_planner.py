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
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            os.environ.setdefault(key, value)
            loaded = True
    return loaded


DEFAULT_SYSTEM_PROMPT = (
    "You are a disciplined ML research assistant for the KuaiRand benchmark. "
    "Your job is to choose the next experiment from the allowed experiment registry. "
    "The history contains past validation metrics; ignore hidden test data. "
    "Prefer an experiment that addresses a likely weakness, keeps training cheap, "
    "and is consistent with the benchmark's goal of improving validation primary score. "
    "Return only valid JSON with a top-level 'proposal' object."
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
    max_tokens: int = 300
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def disabled(cls) -> "LLMPlannerConfig":
        return cls(enabled=False)

    @classmethod
    def from_env(cls) -> "LLMPlannerConfig":
        enabled = os.getenv("KUAI_LLM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        api_key = os.getenv("KUAI_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        return cls(
            enabled=enabled,
            provider=os.getenv("KUAI_LLM_PROVIDER", "openai").strip() or "openai",
            model=os.getenv("KUAI_LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
            api_key=api_key,
            base_url=os.getenv("KUAI_LLM_BASE_URL") or None,
            api_version=os.getenv("KUAI_LLM_API_VERSION") or None,
            deployment=os.getenv("KUAI_LLM_DEPLOYMENT") or None,
            temperature=float(os.getenv("KUAI_LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("KUAI_LLM_MAX_TOKENS", "300")),
            system_prompt=os.getenv("KUAI_LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        )


@dataclass
class ProposedExperiment:
    name: str
    family: str
    params: Dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""
    code_diff: str = ""


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        out: Dict[str, Any] = {}
        for item in value:
            if isinstance(item, dict):
                out.update(item)
        return out
    return {}


def normalize_proposal(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, str):
        candidate = raw.strip()
        if not candidate:
            raise ValueError("Empty proposal string")
        try:
            raw = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Proposal is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Proposal must be a dictionary, got {type(raw).__name__}")

    proposal = raw.get("proposal", raw)
    if not isinstance(proposal, dict):
        raise ValueError("Proposal payload must decode to a dictionary")

    name = str(proposal.get("name") or proposal.get("experiment") or proposal.get("model") or "").strip()
    if not name:
        raise ValueError("Proposal missing a valid 'name' or 'experiment' field")

    params = _coerce_mapping(proposal.get("params") or proposal.get("arguments") or {})
    hypothesis = str(proposal.get("hypothesis") or proposal.get("reason") or "")
    code_diff = str(proposal.get("code_diff") or proposal.get("changes") or "")
    family = str(proposal.get("family") or proposal.get("family_name") or "generic")

    return {
        "name": name,
        "family": family,
        "params": params,
        "hypothesis": hypothesis,
        "code_diff": code_diff,
    }


def resolve_experiment_spec(raw_proposal: Any, registry: Optional[Any] = None) -> ProposedExperiment:
    proposal = normalize_proposal(raw_proposal)
    name = proposal["name"]
    family = proposal["family"]

    if registry is not None:
        names = getattr(registry, "_items", {})
        if name in names:
            family = names[name].family

    return ProposedExperiment(
        name=name,
        family=family,
        params=dict(proposal.get("params") or {}),
        hypothesis=str(proposal.get("hypothesis") or ""),
        code_diff=str(proposal.get("code_diff") or ""),
    )


def _summarize_history(previous_records: Iterable[Dict[str, Any]]) -> str:
    rows: List[Dict[str, Any]] = []
    for record in previous_records:
        metrics = record.get("metrics") or {}
        valid = metrics.get("valid") or {}
        rows.append({
            "experiment": record.get("experiment"),
            "family": record.get("family"),
            "primary": valid.get("primary"),
            "gauc": valid.get("gauc"),
            "ndcg": valid.get("ndcg@5") if "ndcg@5" in valid else valid.get("ndcg"),
            "status": record.get("status"),
            "error": record.get("error"),
        })
    return json.dumps(rows[-10:], ensure_ascii=False)


def summarize_for_prompt(previous_records: Iterable[Dict[str, Any]]) -> str:
    """Create the compact prompt context the LLM can reason over."""
    history = _summarize_history(previous_records)
    return (
        "Recent validation history:\n"
        f"{history}\n\n"
        "Prioritize hypotheses that address where the model is currently weak and keep the next run cheap."
    )


class LLMPlanner:
    def __init__(self, config: Optional[LLMPlannerConfig] = None):
        self.config = config or LLMPlannerConfig.from_env()
        self.last_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and (self.config.api_key or self.config.base_url))

    def _build_payload(self, registry: Any, previous_records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        names = list(getattr(registry, "_items", {}).keys())
        history = summarize_for_prompt(previous_records)
        body = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": json.dumps({
                    "allowed_experiments": names,
                    "history": history,
                    "task": "Choose only one new experiment to run next. Return JSON with a top-level 'proposal' object containing name, family, params, hypothesis, and code_diff. Keep params within the registered experiment defaults and avoid unsupported values.",
                }, ensure_ascii=False)}
            ],
        }
        if self.config.provider.lower() != "azure":
            body["response_format"] = {"type": "json_object"}
        return body

    def _build_request(self) -> tuple[str, Dict[str, str]]:
        provider = (self.config.provider or "openai").lower()
        if self.config.base_url:
            base_url = self.config.base_url.rstrip("/")
        elif provider == "openai":
            base_url = "https://api.openai.com/v1"
        elif provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        elif provider == "azure":
            raise ValueError("Azure requires KUAI_LLM_BASE_URL to be set to your resource endpoint (e.g. https://<resource>.openai.azure.com).")
        else:
            base_url = "https://api.openai.com/v1"

        if provider == "azure":
            if not self.config.api_key:
                raise ValueError("Azure provider requires KUAI_LLM_API_KEY to be set.")
            if not self.config.api_version:
                api_version = "2024-02-01"
            else:
                api_version = self.config.api_version
            deployment = self.config.deployment or self.config.model
            url = f"{base_url.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            headers = {
                "Content-Type": "application/json",
                "api-key": self.config.api_key,
            }
            return url, headers

        if base_url.endswith("/chat/completions"):
            url = base_url
        else:
            url = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        return url, headers

    def _call_provider(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("LLM planner is disabled or missing API configuration")

        if not self.config.api_key:
            raise RuntimeError("LLM planner requires KUAI_LLM_API_KEY or OPENAI_API_KEY to be set.")

        url, headers = self._build_request()
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except error.HTTPError as exc:
            try:
                details = exc.read().decode("utf-8", errors="replace")
            except Exception:
                details = str(exc)
            raise RuntimeError(f"LLM provider request failed: {details}") from exc

        message = data.get("choices", [{}])[0].get("message", {})
        content = message.get("content") or data.get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str):
            content = str(content)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end + 1])
            else:
                raise ValueError(f"LLM response was not JSON: {content[:400]}")
        return parsed

    def suggest_next_experiment(self, registry: Any, previous_records: Iterable[Dict[str, Any]]) -> Optional[ProposedExperiment]:
        if not self.enabled:
            return None

        try:
            payload = self._build_payload(registry, previous_records)
            response = self._call_provider(payload)
            normalized = normalize_proposal(response)
            spec = resolve_experiment_spec(normalized, registry)
            self.last_error = None
            return spec
        except Exception as exc:  # pragma: no cover - runtime external API path
            self.last_error = str(exc)
            return None
