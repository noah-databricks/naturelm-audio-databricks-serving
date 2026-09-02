"""Offline MLflow adapter for EarthSpeciesProject/NatureLM-audio."""

from __future__ import annotations

import base64
import io
import threading
import time
from pathlib import Path

from mlflow.pyfunc import PythonModel


class NatureLMAudioModel(PythonModel):
    """Serve compressed audio bytes and a question through NatureLM-audio."""

    def load_context(self, context):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("NatureLM-audio requires a GPU serving workload.")

        from NatureLM.infer import Pipeline
        from NatureLM.models import NatureLM
        from NatureLM.models.Qformer import BertConfig

        cfg = context.model_config or {}
        self.max_audio_bytes = int(cfg.get("max_audio_bytes", 20 * 1024 * 1024))
        self.max_audio_seconds = float(cfg.get("max_audio_seconds", 60.0))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 64))
        self.device = "cuda:0"
        self._predict_lock = threading.Lock()

        # NatureLM hard-codes BertConfig.from_pretrained("bert-base-uncased") only
        # to build its randomly initialized two-layer Q-Former. Redirect that
        # config read to the packaged file so startup is completely offline.
        bert_config_path = str(Path(context.artifacts["bert_config"]))
        BertConfig.from_pretrained = classmethod(
            lambda cls, *_args, **_kwargs: cls.from_json_file(bert_config_path)
        )

        # Override llama_path from NatureLM's config.json. Both snapshots are
        # immutable MLflow artifacts, so no Hugging Face token or Volume mount is
        # needed by the serving container.
        self.model = NatureLM.from_pretrained(
            context.artifacts["naturelm_snapshot"],
            llama_path=Path(context.artifacts["llama_snapshot"]),
            local_files_only=True,
            map_location="cpu",
            # The upstream CPU branch loads Llama in float32 before moving the
            # composite model, which doubles peak memory for this 8B model.
            # Selecting CUDA preserves NatureLM's bfloat16 load path; its
            # remaining audio modules are transferred by the .to() below.
            device=self.device,
        )
        self.model = self.model.eval().to(self.device)
        self.model.llama_tokenizer.pad_token_id = self.model.llama_tokenizer.eos_token_id
        self.model.llama_model.generation_config.pad_token_id = (
            self.model.llama_tokenizer.pad_token_id
        )

        self.pipeline = Pipeline(
            model=self.model,
            cfg_path=context.artifacts["inference_config"],
        )
        self.pipeline.cfg.generate.max_new_tokens = self.max_new_tokens

    def _decode_audio(self, encoded: str):
        import numpy as np
        import soundfile as sf

        if not encoded:
            raise ValueError("audio_base64 must not be empty.")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("audio_base64 is not valid base64.") from exc
        if len(raw) > self.max_audio_bytes:
            raise ValueError(
                f"Decoded audio exceeds the {self.max_audio_bytes}-byte request limit."
            )

        try:
            audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
        except Exception as exc:
            raise ValueError(
                "Audio could not be decoded; send WAV, FLAC, OGG, or MP3 bytes."
            ) from exc
        if sample_rate <= 0 or not getattr(audio, "size", 0):
            raise ValueError("Decoded audio is empty or has an invalid sample rate.")
        if not np.isfinite(audio).all():
            raise ValueError("Decoded audio contains non-finite samples.")

        duration = float(audio.shape[0]) / float(sample_rate)
        if duration < 0.5:
            raise ValueError("Audio must be at least 0.5 seconds long.")
        if duration > self.max_audio_seconds:
            raise ValueError(
                f"Audio is {duration:.2f}s; the configured limit is "
                f"{self.max_audio_seconds:.2f}s."
            )
        return audio, int(sample_rate), duration

    def predict(self, context, model_input, params=None):
        import pandas as pd

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        required = {"audio_base64", "query"}
        missing = sorted(required.difference(model_input.columns))
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")

        outputs = []
        for row in model_input.to_dict(orient="records"):
            query = str(row["query"]).strip()
            if not query or query.lower() == "nan":
                raise ValueError("query must be a non-empty string.")
            audio, sample_rate, duration = self._decode_audio(str(row["audio_base64"]))

            started = time.monotonic()
            # Serialize inference within a replica. Model Serving scales replicas
            # for concurrency; the upstream Pipeline mutates/caches model state.
            with self._predict_lock:
                result = self.pipeline(
                    audios=[audio],
                    queries=[query],
                    input_sample_rate=sample_rate,
                    window_length_seconds=10.0,
                    hop_length_seconds=10.0,
                )[0]
            outputs.append(
                {
                    "result": result,
                    "query": query,
                    "audio_duration_seconds": round(duration, 3),
                    "sample_rate": sample_rate,
                    "inference_seconds": round(time.monotonic() - started, 3),
                    "device": self.device,
                }
            )
        return pd.DataFrame(outputs)


import mlflow

mlflow.models.set_model(NatureLMAudioModel())
