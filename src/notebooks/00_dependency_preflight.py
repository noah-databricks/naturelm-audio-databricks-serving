# Databricks notebook source
# MAGIC %md
# MAGIC # NatureLM-audio dependency preflight
# MAGIC
# MAGIC This lightweight task imports the exact packages and vendored NatureLM
# MAGIC modules used by Model Serving before the large MLflow artifact is created.

# COMMAND ----------

import importlib.metadata
import json
import sys
from pathlib import Path

dbutils.widgets.text("naturelm_code_path", "")
naturelm_package = Path(dbutils.widgets.get("naturelm_code_path"))
sys.path.insert(0, str(naturelm_package.parent))

import peft
import torch
import torchaudio
import torchvision
import transformers
from NatureLM.infer import Pipeline
from NatureLM.models import NatureLM

versions = {
    name: importlib.metadata.version(name)
    for name in (
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "peft",
        "safetensors",
    )
}
assert versions["torch"].split("+")[0] == "2.7.1"
assert versions["torchaudio"].split("+")[0] == "2.7.1"
assert versions["torchvision"].split("+")[0] == "0.22.1"
print(json.dumps(versions, indent=2))
dbutils.notebook.exit(json.dumps({"status": "PASSED", "versions": versions}))
