"""Shared training configuration objects."""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class EndpointConfig:
    api_base_url: str
    api_key: str

    @classmethod
    def from_env(cls, api_base_env: str = "API_BASE_URL", api_key_env: str = "HF_TOKEN") -> "EndpointConfig":
        api_base_url = os.getenv(api_base_env) or "https://router.huggingface.co/v1"
        api_key = os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
        if not api_key:
            raise SystemExit(
                "Missing API key. Set HF_TOKEN, OPENAI_API_KEY, or API_KEY before running the training pipeline."
            )
        return cls(api_base_url=api_base_url, api_key=api_key)


@dataclass
class SimulationConfig:
    policy_model: str
    critic_model: str
    dataset_name: str | None = None
    dataset_split: str = "train"
    max_seed_examples: int | None = None
    episodes_per_seed: int = 2
    num_candidates: int = 4
    rollout_steps: int = 6
    max_turns: int = 16
    max_completion_tokens: int = 220
    temperature: float = 0.7
    rollout_temperature: float = 0.3
    max_recent_turns_in_prompt: int = 8


@dataclass
class RewardModelConfig:
    model_name: str
    output_dir: str
    max_length: int = 1024
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    num_train_epochs: float = 1.0
    eval_ratio: float = 0.1
    freeze_backbone: bool = True


@dataclass
class GRPOTrainingConfig:
    model_name: str
    reward_model_dir: str
    output_dir: str
    learning_rate: float = 1e-6
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 4
    max_prompt_length: int = 1024
    max_completion_length: int = 256
    max_steps: int = 100
    temperature: float = 0.8
    think_format_weight: float = 0.2
