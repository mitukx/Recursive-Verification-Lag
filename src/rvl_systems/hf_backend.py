from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

from .precision import resolve_precision_name, torch_dtype_for_name
from .types import Generation


@dataclass
class HFLocalBackend:
    """Local Hugging Face causal-LM rollout backend with token log-prob capture."""

    model_name: str
    max_new_tokens: int = 64
    device: str | None = None
    precision: str = "auto"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: str | None = self.device
        self._resolved_precision: str | None = None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HFLocalBackend requires optional systems dependencies; "
                "install requirements-systems.txt"
            ) from exc

        if self._device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

        bf16_supported = bool(
            self._device == "cuda"
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        self._resolved_precision = resolve_precision_name(
            self._device,
            self.precision,
            bf16_supported=bf16_supported,
        )
        dtype = torch_dtype_for_name(torch, self._resolved_precision)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            dtype=dtype,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer must define eos_token_id or pad_token_id")
            tokenizer.pad_token = tokenizer.eos_token
        model.to(self._device)
        model.train()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    @property
    def model(self) -> Any:
        self.ensure_loaded()
        return self._model

    @property
    def tokenizer(self) -> Any:
        self.ensure_loaded()
        return self._tokenizer

    @property
    def resolved_device(self) -> str:
        self.ensure_loaded()
        assert self._device is not None
        return self._device

    @property
    def resolved_precision(self) -> str:
        self.ensure_loaded()
        assert self._resolved_precision is not None
        return self._resolved_precision

    async def generate(
        self,
        prompt_id: str,
        prompt: str,
        *,
        n: int,
        temperature: float,
        seed: int,
    ) -> list[Generation]:
        return await asyncio.to_thread(
            self._generate_sync,
            prompt_id,
            prompt,
            n,
            temperature,
            seed,
        )

    def _generate_sync(
        self,
        prompt_id: str,
        prompt: str,
        n: int,
        temperature: float,
        seed: int,
    ) -> list[Generation]:
        if n <= 0:
            raise ValueError("n must be positive")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if temperature == 0 and n != 1:
            raise ValueError("greedy decoding supports exactly one return sequence")
        self.ensure_loaded()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        device = self.resolved_device

        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        input_length = int(encoded["input_ids"].shape[1])
        prompt_token_ids = encoded["input_ids"][0].tolist()

        start = time.perf_counter()
        was_training = model.training
        model.eval()
        from transformers import GenerationConfig
        # A pretrained chat model can inherit repetition penalties, suppression,
        # or truncation processors. Their scores are not the policy logits the
        # learner evaluates. Start from neutral defaults, retaining only special
        # token identities, and make temperature the sole sampling transform.
        special = model.generation_config
        generation_config = GenerationConfig(
            bos_token_id=special.bos_token_id,
            eos_token_id=special.eos_token_id if special.eos_token_id is not None else tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_k=0,
            top_p=1.0,
        )
        # Expand neutral fields as explicit overrides: the supported, pinned
        # Transformers release can otherwise merge model defaults even into a
        # fresh GenerationConfig. Do not mix that object with generation kwargs.
        generation_kwargs = {
            key: value for key, value in generation_config.to_dict().items()
            if not key.startswith("_") and key not in {"transformers_version", "max_length"}
        }
        generation_kwargs.update({
            "do_sample": temperature > 0,
            "num_return_sequences": n,
            "max_new_tokens": self.max_new_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": tokenizer.pad_token_id,
        })
        if temperature > 0:
            generation_kwargs.update(
                {
                    "temperature": temperature,
                    "top_k": 0,
                    "top_p": 1.0,
                }
            )
        try:
            with torch.inference_mode():
                outputs = model.generate(
                    **encoded,
                    **generation_kwargs,
                )
                transition = model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    normalize_logits=True,
                )
        finally:
            model.train(was_training)
        latency = time.perf_counter() - start

        generations: list[Generation] = []
        eos = generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
        pad_id = tokenizer.pad_token_id
        for row in range(outputs.sequences.shape[0]):
            raw_ids = outputs.sequences[row, input_length:].tolist()
            response_ids: list[int] = []
            for token_id in raw_ids:
                if token_id == pad_id and pad_id not in eos_ids:
                    break
                response_ids.append(int(token_id))
                if token_id in eos_ids:
                    break
            token_logprobs = [
                float(x)
                for x in transition[row, : len(response_ids)].detach().cpu().tolist()
            ]
            text = tokenizer.decode(response_ids, skip_special_tokens=True)
            generations.append(
                Generation(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    response=text,
                    logprob=sum(token_logprobs),
                    token_count=len(response_ids),
                    latency_s=latency,
                    metadata={
                        "backend": "hf-local",
                        "model": self.model_name,
                        "device": device,
                        "precision": self.resolved_precision,
                        "sampling_temperature": temperature,
                        "logprob_distribution": "neutral_temperature_scaled_policy" if temperature > 0 else "greedy_scores_not_trainable",
                        "prompt_token_ids": prompt_token_ids,
                        "response_token_ids": response_ids,
                        "response_token_logprobs": token_logprobs,
                    },
                )
            )
        return generations
