# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Explicitly deploy and validate NatureLM-audio
# MAGIC
# MAGIC The default is **NO DEPLOYMENT**. Review the resolved configuration, then
# MAGIC set `confirm_deploy` to `YES` before running the deployment cell.
# MAGIC
# MAGIC GPU Model Serving is billable while active. Scale-to-zero is enabled.

# COMMAND ----------

import json
import re
import time
from pathlib import Path

from mlflow.deployments import get_deploy_client
from mlflow.tracking import MlflowClient

WIDGET_DEFAULTS = {
    "catalog": "YOUR_CATALOG",
    "schema": "YOUR_SCHEMA",
    "model_name": "naturelm_audio",
    "endpoint_name": "naturelm-audio-serving",
    "workload_type": "GPU_MEDIUM",
    "workload_size": "Small",
    "confirm_deploy": "NO",
    "smoke_test_audio_base64_path": "",
    "smoke_test_query": "What is the common name for the focal species in the audio? Answer:",
    "expected_answer_substring": "Green Treefrog",
}
for name, default in WIDGET_DEFAULTS.items():
    dbutils.widgets.text(name, default)

cfg = {name: dbutils.widgets.get(name).strip() for name in WIDGET_DEFAULTS}
full_model_name = f'{cfg["catalog"]}.{cfg["schema"]}.{cfg["model_name"]}'

registry = MlflowClient(registry_uri="databricks-uc")
version = str(registry.get_model_version_by_alias(full_model_name, "prod").version)
model_version = registry.get_model_version(full_model_name, version)
if model_version.status != "READY":
    raise RuntimeError(f"Model version {full_model_name}/{version} is not READY")
served_entity_name = re.sub(r"[^A-Za-z0-9_]", "_", f'{cfg["model_name"]}_{version}')

deployment_preview = {
    "endpoint_name": cfg["endpoint_name"],
    "registered_model": full_model_name,
    "model_version": version,
    "served_entity_name": served_entity_name,
    "workload_type": cfg["workload_type"],
    "workload_size": cfg["workload_size"],
    "scale_to_zero_enabled": True,
}
print(json.dumps(deployment_preview, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Deployment cell — explicit confirmation required
# MAGIC
# MAGIC Set the `confirm_deploy` widget to `YES`, then run this cell and those below.

# COMMAND ----------

if cfg["confirm_deploy"] != "YES":
    dbutils.notebook.exit(
        json.dumps(
            {
                "status": "NOT_DEPLOYED",
                "message": "Set confirm_deploy=YES after reviewing the configuration.",
                "deployment_preview": deployment_preview,
            }
        )
    )

deploy = get_deploy_client("databricks")
entity = {
    "name": served_entity_name,
    "entity_name": full_model_name,
    "entity_version": version,
    "workload_type": cfg["workload_type"],
    "workload_size": cfg["workload_size"],
    "scale_to_zero_enabled": True,
}
endpoint_config = {
    "served_entities": [entity],
    "traffic_config": {
        "routes": [{"served_model_name": served_entity_name, "traffic_percentage": 100}]
    },
}

try:
    existing_endpoint = deploy.get_endpoint(endpoint=cfg["endpoint_name"])
except Exception as exc:
    if "RESOURCE_DOES_NOT_EXIST" not in str(exc) and "404" not in str(exc):
        raise
    deploy.create_endpoint(name=cfg["endpoint_name"], config=endpoint_config)
    deployment_action = "created"
else:
    def is_desired(entities):
        candidate = entities[0] if len(entities) == 1 else {}
        return (
            candidate.get("entity_name") == entity["entity_name"]
            and str(candidate.get("entity_version")) == entity["entity_version"]
            and candidate.get("name") == entity["name"]
            and candidate.get("workload_type") == entity["workload_type"]
            and candidate.get("workload_size") == entity["workload_size"]
            and candidate.get("scale_to_zero_enabled") is True
        )

    current_entities = existing_endpoint.get("config", {}).get("served_entities", [])
    pending_entities = existing_endpoint.get("pending_config", {}).get(
        "served_entities", []
    )
    if is_desired(current_entities):
        deployment_action = "unchanged"
        print("Endpoint already has the desired model and compute; skipping update.")
    elif is_desired(pending_entities):
        deployment_action = "update_in_progress"
        print("The desired configuration is already being deployed; waiting for it.")
    elif existing_endpoint.get("state", {}).get("config_update") == "IN_PROGRESS":
        raise RuntimeError(
            "The endpoint is already deploying a different configuration. "
            "Wait for that update to finish, then rerun this confirmed job."
        )
    else:
        deploy.update_endpoint(endpoint=cfg["endpoint_name"], config=endpoint_config)
        deployment_action = "updated"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Wait for this configuration to become ready

# COMMAND ----------

deadline = time.monotonic() + 60 * 90
last_state = None
while time.monotonic() < deadline:
    endpoint = deploy.get_endpoint(endpoint=cfg["endpoint_name"])
    state = endpoint.get("state", {})
    current_state = (state.get("ready"), state.get("config_update"))
    if current_state != last_state:
        print({"ready": current_state[0], "config_update": current_state[1]})
        last_state = current_state
    if current_state[0] == "READY" and current_state[1] in ("NOT_UPDATING", None):
        break
    if current_state[1] == "UPDATE_FAILED":
        raise RuntimeError(json.dumps(endpoint, indent=2, default=str))
    time.sleep(20)
else:
    raise TimeoutError("Endpoint did not finish provisioning within 90 minutes")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Real audio smoke test

# COMMAND ----------

audio_base64 = Path(cfg["smoke_test_audio_base64_path"]).read_text().replace("\n", "").strip()
request = {
    "dataframe_records": [
        {
            "audio_base64": audio_base64,
            "query": cfg["smoke_test_query"],
        }
    ]
}
started = time.monotonic()
response = deploy.predict(endpoint=cfg["endpoint_name"], inputs=request)
request_latency_seconds = time.monotonic() - started
print(json.dumps(response, indent=2, default=str))

predictions = response.get("predictions") or []
if len(predictions) != 1:
    raise AssertionError(f"Expected one prediction, received {len(predictions)}")
prediction = predictions[0]
for required_key in (
    "result",
    "query",
    "audio_duration_seconds",
    "sample_rate",
    "inference_seconds",
    "device",
):
    if required_key not in prediction:
        raise AssertionError(f"Prediction is missing output field {required_key!r}")
if prediction["device"] != "cuda:0":
    raise AssertionError(f"Model did not report GPU inference: {prediction['device']!r}")
if cfg["expected_answer_substring"].lower() not in prediction["result"].lower():
    raise AssertionError(
        f"Expected {cfg['expected_answer_substring']!r} in model output: "
        f"{prediction['result']!r}"
    )

result = {
    **deployment_preview,
    "ready": True,
    "deployment_action": deployment_action,
    "request_latency_seconds": round(request_latency_seconds, 3),
    "semantic_smoke_test": "PASSED",
    "smoke_test_response": response,
    "cold_start_note": (
        "After at least 30 traffic-free minutes, rerun this confirmed job. "
        "An unchanged endpoint skips the update and measures cold-start latency."
    ),
}
dbutils.notebook.exit(json.dumps(result, default=str))
