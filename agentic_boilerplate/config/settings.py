"""AgentConfig and LLMConfig dataclasses with YAML/JSON loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for the language model backend."""

    provider: str = "stub"  # "stub", "openai", "anthropic"
    model: str = "stub-model"
    api_key: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7

    @classmethod
    def from_dict(cls, data: dict) -> "LLMConfig":
        return cls(
            provider=data.get("provider", "stub"),
            model=data.get("model", "stub-model"),
            api_key=data.get("api_key"),
            max_tokens=int(data.get("max_tokens", 4096)),
            temperature=float(data.get("temperature", 0.7)),
        )

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


@dataclass
class AgentConfig:
    """Top-level agent configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    max_context_messages: int = 100
    compact_threshold: int = 80
    yolo_mode: bool = False
    system_prompt: str = "You are a helpful assistant."

    @classmethod
    def from_dict(cls, data: dict) -> "AgentConfig":
        llm_data = data.get("llm", {})
        return cls(
            llm=LLMConfig.from_dict(llm_data),
            max_context_messages=int(data.get("max_context_messages", 100)),
            compact_threshold=int(data.get("compact_threshold", 80)),
            yolo_mode=bool(data.get("yolo_mode", False)),
            system_prompt=data.get("system_prompt", "You are a helpful assistant."),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        import yaml  # type: ignore[import-untyped]

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "AgentConfig":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return {
            "llm": self.llm.to_dict(),
            "max_context_messages": self.max_context_messages,
            "compact_threshold": self.compact_threshold,
            "yolo_mode": self.yolo_mode,
            "system_prompt": self.system_prompt,
        }
