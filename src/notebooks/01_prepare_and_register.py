# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Validate, package, and register NatureLM-audio
# MAGIC
# MAGIC This notebook validates the NatureLM-audio fine-tune snapshot and the
# MAGIC authorized Llama 3.1 8B Instruct base snapshot already present in Unity
# MAGIC Catalog Volumes. It packages both, together with pinned inference code,
# MAGIC into one immutable MLflow model and assigns the registered version `@prod`.
# MAGIC
# MAGIC It does **not** create or update a serving endpoint.

# COMMAND ----------

import hashlib
import importlib.metadata
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from safetensors import safe_open

WIDGET_DEFAULTS = {
    "catalog": "YOUR_CATALOG",
    "schema": "YOUR_SCHEMA",
    "model_name": "naturelm_audio",
    "naturelm_model_path": "/Volumes/YOUR_CATALOG/YOUR_SCHEMA/YOUR_VOLUME/naturelm_audio",
    "llama_model_path": "/Volumes/YOUR_CATALOG/YOUR_SCHEMA/YOUR_VOLUME/llama_3_1_8b_instruct",
    "validation_naturelm_repo_id": "",
    "validation_naturelm_revision": "",
    "validation_llama_repo_id": "",
    "validation_llama_revision": "",
    "max_audio_seconds": "60",
    "max_new_tokens": "64",
    "experiment_id": "",
    "model_code_path": "",
    "naturelm_code_path": "",
    "bert_config_path": "",
    "inference_config_path": "",
}
for name, default in WIDGET_DEFAULTS.items():
    dbutils.widgets.text(name, default)

cfg = {name: dbutils.widgets.get(name).strip() for name in WIDGET_DEFAULTS}
cfg["max_audio_seconds"] = float(cfg["max_audio_seconds"])
cfg["max_new_tokens"] = int(cfg["max_new_tokens"])

identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
for key in ("catalog", "schema", "model_name"):
    if not identifier.fullmatch(cfg[key]):
        raise ValueError(f"{key} must be an unquoted Unity Catalog identifier: {cfg[key]!r}")
for key in ("naturelm_model_path", "llama_model_path"):
    if not cfg[key].startswith("/Volumes/"):
        raise ValueError(f"{key} must be an existing Unity Catalog /Volumes/... directory")
for repo_key, revision_key in (
    ("validation_naturelm_repo_id", "validation_naturelm_revision"),
    ("validation_llama_repo_id", "validation_llama_revision"),
):
    if cfg[repo_key] and not re.fullmatch(r"[0-9a-f]{40}", cfg[revision_key]):
        raise ValueError(f"{revision_key} must be an immutable 40-character commit SHA")
if not (1 <= cfg["max_new_tokens"] <= 300):
    raise ValueError("max_new_tokens must be between 1 and 300")
if not (0.5 <= cfg["max_audio_seconds"] <= 600):
    raise ValueError("max_audio_seconds must be between 0.5 and 600")

naturelm_path = Path(cfg["naturelm_model_path"])
llama_path = Path(cfg["llama_model_path"])
full_model_name = f'{cfg["catalog"]}.{cfg["schema"]}.{cfg["model_name"]}'
print(
    json.dumps(
        {
            "registered_model": full_model_name,
            "naturelm_model_path": str(naturelm_path),
            "llama_model_path": str(llama_path),
            "endpoint_created": False,
        },
        indent=2,
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Optional, validation-target-only download
# MAGIC
# MAGIC Customer runs leave both repository IDs empty and only read the supplied
# MAGIC Volume directories. The internal validation target downloads immutable
# MAGIC public revisions into a managed Volume.

# COMMAND ----------

from huggingface_hub import snapshot_download

download_specs = (
    (
        cfg["validation_naturelm_repo_id"],
        cfg["validation_naturelm_revision"],
        naturelm_path,
    ),
    (
        cfg["validation_llama_repo_id"],
        cfg["validation_llama_revision"],
        llama_path,
    ),
)
for repo_id, revision, local_path in download_specs:
    if repo_id:
        local_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_path),
            max_workers=8,
            ignore_patterns=["original/*", "*.pth", "*.bin"],
        )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Fail-fast structure, tensor-header, and SHA-256 validation

# COMMAND ----------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_snapshot(path: Path, kind: str) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(f"{kind} snapshot directory does not exist: {path}")
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and "/.cache/" not in str(item)
    )
    relative_files = [str(item.relative_to(path)) for item in files]
    tensor_files = [item for item in files if item.suffix == ".safetensors"]

    required = ["config.json"]
    if kind == "NatureLM-audio":
        required.append("model.safetensors")
    else:
        required.append("model.safetensors.index.json")
    missing = [name for name in required if name not in relative_files]
    has_tokenizer = any(
        name in relative_files
        for name in ("tokenizer.json", "tokenizer.model", "original/tokenizer.model")
    )
    if kind == "Llama 3.1 8B Instruct" and not has_tokenizer:
        missing.append("tokenizer files")
    if missing or not tensor_files:
        raise ValueError(
            f"{kind} is not a complete expected snapshot: "
            f"missing={missing}, safetensors={len(tensor_files)}"
        )

    def inspect_tensor_file(item: Path) -> dict:
        with safe_open(str(item), framework="numpy", device="cpu") as tensors:
            keys = list(tensors.keys())
        return {
            "path": str(item.relative_to(path)),
            "bytes": item.stat().st_size,
            "sha256": sha256(item),
            "tensor_count": len(keys),
        }

    with ThreadPoolExecutor(max_workers=min(8, len(tensor_files))) as pool:
        tensor_details = list(pool.map(inspect_tensor_file, tensor_files))
    with (path / "config.json").open(encoding="utf-8") as handle:
        model_config = json.load(handle)
    return {
        "kind": kind,
        "path": str(path),
        "architecture": model_config.get("architectures"),
        "model_type": model_config.get("model_type"),
        "total_bytes": sum(item.stat().st_size for item in files),
        "file_count": len(files),
        "safetensors": tensor_details,
    }


naturelm_manifest = validate_snapshot(naturelm_path, "NatureLM-audio")
llama_manifest = validate_snapshot(llama_path, "Llama 3.1 8B Instruct")

# The public validation mirror is accepted only when every weight shard is
# byte-identical to Meta's official repository metadata.
OFFICIAL_LLAMA_WEIGHT_HASHES = {
    "model-00001-of-00004.safetensors": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
    "model-00002-of-00004.safetensors": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
    "model-00003-of-00004.safetensors": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
    "model-00004-of-00004.safetensors": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
}
if cfg["validation_llama_repo_id"]:
    actual = {item["path"]: item["sha256"] for item in llama_manifest["safetensors"]}
    if actual != OFFICIAL_LLAMA_WEIGHT_HASHES:
        raise ValueError("Validation Llama weights do not match Meta's official SHA-256 values")

manifest = {
    "created_at_epoch_seconds": int(time.time()),
    "naturelm_repo_id": cfg["validation_naturelm_repo_id"] or "customer-volume",
    "naturelm_revision": cfg["validation_naturelm_revision"] or "customer-managed",
    "llama_repo_id": cfg["validation_llama_repo_id"] or "customer-volume",
    "llama_revision": cfg["validation_llama_revision"] or "customer-managed",
    "naturelm_source_commit": "c708df7a4cc294ca8d4aaf0498794b5674ce20b1",
    "snapshots": [naturelm_manifest, llama_manifest],
}
manifest_path = Path("/tmp/naturelm_audio_integrity_manifest.json")
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Log and register one offline, immutable MLflow model
# MAGIC
# MAGIC The model artifact intentionally includes both snapshots. Endpoint startup
# MAGIC therefore has no dependency on the external Volume, S3, Hugging Face, or a
# MAGIC long-lived Hugging Face token.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
if cfg["experiment_id"]:
    mlflow.set_experiment(experiment_id=cfg["experiment_id"])

runtime_mlflow = importlib.metadata.version("mlflow")
serving_requirements = [
    f"mlflow=={runtime_mlflow}",
    "torch==2.7.1",
    "torchaudio==2.7.1",
    "torchvision==0.22.1",
    "transformers[sentencepiece]==4.57.3",
    "accelerate==1.12.0",
    "safetensors==0.7.0",
    "huggingface-hub==0.36.0",
    "peft==0.17.1",
    "pandas==2.3.3",
    "numpy==2.2.6",
    "scipy==1.15.3",
    "soundfile==0.13.1",
    "resampy==0.4.3",
    "pydantic==2.12.5",
    "pydantic-settings==2.12.0",
    "PyYAML==6.0.3",
    "cloudpathlib[gs]==0.23.0",
]

input_example = pd.DataFrame(
    {
        "audio_base64": ["UklGRg=="],
        "query": ["What is the common name for the focal species in the audio? Answer:"],
    }
)
output_example = pd.DataFrame(
    {
        "result": ["#0.00s - 10.00s#: Green Treefrog\n"],
        "query": ["What is the common name for the focal species in the audio? Answer:"],
        "audio_duration_seconds": [10.0],
        "sample_rate": [16000],
        "inference_seconds": [1.0],
        "device": ["cuda:0"],
    }
)

with mlflow.start_run(run_name=f'{cfg["model_name"]}-package') as run:
    mlflow.log_params(
        {
            "naturelm_model_path": cfg["naturelm_model_path"],
            "llama_model_path": cfg["llama_model_path"],
            "naturelm_revision": manifest["naturelm_revision"],
            "llama_revision": manifest["llama_revision"],
            "naturelm_bytes": naturelm_manifest["total_bytes"],
            "llama_bytes": llama_manifest["total_bytes"],
            "max_audio_seconds": cfg["max_audio_seconds"],
            "max_new_tokens": cfg["max_new_tokens"],
        }
    )
    mlflow.log_artifact(str(manifest_path), artifact_path="integrity")
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=cfg["model_code_path"],
        artifacts={
            "naturelm_snapshot": str(naturelm_path),
            "llama_snapshot": str(llama_path),
            "bert_config": cfg["bert_config_path"],
            "inference_config": cfg["inference_config_path"],
        },
        code_paths=[cfg["naturelm_code_path"]],
        model_config={
            "max_audio_bytes": 20 * 1024 * 1024,
            "max_audio_seconds": cfg["max_audio_seconds"],
            "max_new_tokens": cfg["max_new_tokens"],
        },
        input_example=input_example,
        signature=infer_signature(input_example, output_example),
        pip_requirements=serving_requirements,
        registered_model_name=full_model_name,
        metadata={
            "model_family": "EarthSpeciesProject/NatureLM-audio",
            "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "naturelm_license": "CC-BY-NC-SA-4.0",
            "naturelm_source_commit": manifest["naturelm_source_commit"],
            "offline_serving": True,
        },
    )

version = str(info.registered_model_version)
client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(full_model_name, "prod", version)
client.set_model_version_tag(full_model_name, version, "model_family", "NatureLM-audio")
client.set_model_version_tag(full_model_name, version, "offline_serving", "true")
client.set_model_version_tag(
    full_model_name,
    version,
    "integrity_manifest",
    "integrity/naturelm_audio_integrity_manifest.json",
)

result = {
    "run_id": run.info.run_id,
    "registered_model": full_model_name,
    "version": version,
    "alias": "prod",
    "naturelm_bytes": naturelm_manifest["total_bytes"],
    "llama_bytes": llama_manifest["total_bytes"],
    "combined_source_bytes": (
        naturelm_manifest["total_bytes"] + llama_manifest["total_bytes"]
    ),
    "endpoint_created": False,
}
dbutils.notebook.exit(json.dumps(result))
