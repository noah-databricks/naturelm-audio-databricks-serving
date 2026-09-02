# Serve NatureLM-audio from Unity Catalog Volumes

This Databricks bundle registers
[`EarthSpeciesProject/NatureLM-audio`](https://huggingface.co/EarthSpeciesProject/NatureLM-audio)
in MLflow and serves it on a scale-to-zero GPU endpoint.

The NatureLM repository contains about 1.56 GB of audio encoder, Q-Former, and
LoRA weights. A working model also needs the roughly 16 GB
`meta-llama/Meta-Llama-3.1-8B-Instruct` base snapshot. This bundle packages both
Volume directories and the pinned NatureLM inference code into one immutable,
offline MLflow model version.

Endpoint creation is a separate, explicit step.

## Before you start

You need:

- Databricks CLI 0.294 or newer, authenticated to the target workspace.
- An existing Unity Catalog catalog and schema.
- A Volume directory containing the complete NatureLM-audio repository snapshot.
- A second Volume directory containing the complete, authorized Meta Llama 3.1
  8B Instruct repository snapshot.
- `USE CATALOG`, `USE SCHEMA`, `READ VOLUME`, and permission to create a
  registered model and a Model Serving endpoint.

The bundle only reads the supplied Volume paths. It never creates, changes, or
deletes an external location, storage credential, external Volume, S3 object, or
source model file.

AWS Sydney (`ap-southeast-2`) supports custom GPU Model Serving and
scale-to-zero. This workflow does not require a private preview. `GPU_MEDIUM`
uses one 24 GB A10G and is the validated starting point.

## License check

Before deploying, confirm that the intended use complies with all upstream
licenses:

- NatureLM-audio model weights: CC-BY-NC-SA-4.0, including a non-commercial
  restriction.
- NatureLM-audio source code: Apache-2.0.
- Llama 3.1 8B Instruct: Meta Llama 3.1 Community License and Acceptable Use
  Policy.

This bundle does not redistribute either set of weights.

## 1. Set the customer values

From this directory, replace the six example values:

```bash
CUSTOMER_VARS="catalog=YOUR_CATALOG,schema=YOUR_SCHEMA,naturelm_model_path=/Volumes/YOUR_CATALOG/YOUR_SCHEMA/YOUR_VOLUME/naturelm_audio,llama_model_path=/Volumes/YOUR_CATALOG/YOUR_SCHEMA/YOUR_VOLUME/llama_3_1_8b_instruct,model_name=naturelm_audio,endpoint_name=naturelm-audio-serving"
```

The NatureLM path must contain `config.json` and `model.safetensors`. The
Llama path must contain `config.json`, `model.safetensors.index.json`, all
four safetensors shards, and tokenizer files. Point to the resolved snapshot
directories, not the parent Hugging Face cache directory.

## 2. Validate and deploy the bundle

```bash
databricks bundle validate --strict -t customer --var="$CUSTOMER_VARS"
databricks bundle deploy -t customer --var="$CUSTOMER_VARS"
```

This creates two jobs, an MLflow experiment, and an empty Unity Catalog
registered-model resource. It does not create an endpoint.

## 3. Package and register the model

```bash
databricks bundle run prepare_and_register -t customer --var="$CUSTOMER_VARS"
```

The job:

1. validates both repository layouts;
2. reads every safetensors header and computes SHA-256 hashes;
3. records an integrity manifest in the MLflow run;
4. copies both snapshots and pinned inference code into one MLflow artifact;
5. registers a new Unity Catalog model version and assigns the `@prod` alias.

The deliberate weight duplication makes the registered version reproducible and
removes serving-time dependencies on the Volume, S3, Hugging Face, or long-lived
Hugging Face credentials.

## 4. Explicitly deploy the endpoint

First run the command without confirmation to preview the resolved model,
version, endpoint, and GPU configuration:

```bash
databricks bundle run deploy_and_validate -t customer --var="$CUSTOMER_VARS"
```

It exits with `NOT_DEPLOYED` and makes no endpoint change. After review:

```bash
databricks bundle run deploy_and_validate -t customer --var="$CUSTOMER_VARS" --params="confirm_deploy=YES"
```

The confirmed job creates or updates the endpoint, waits for the exact
configuration to become ready, and runs a real audio test whose expected answer
is `Green Treefrog`. Re-running it is safe: if the endpoint already serves the
same model version and compute, the update is skipped.

You can instead open `src/notebooks/02_deploy_and_validate` in the deployed
bundle, review the configuration cell, set the `confirm_deploy` widget to
`YES`, and run the remaining cells interactively.

## Invoke the endpoint

Requests contain compressed audio-file bytes as base64 rather than a very large
JSON array of floating-point samples:

```bash
curl -X POST \
  "https://YOUR_WORKSPACE/serving-endpoints/naturelm-audio-serving/invocations" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "dataframe_records": [{
    "audio_base64": "$(base64 < recording.mp3 | tr -d '\n')",
    "query": "What is the common name for the focal species in the audio? Answer:"
  }]
}
JSON
```

WAV, FLAC, OGG, and MP3 are accepted. Defaults cap decoded files at 20 MB,
audio at 60 seconds, and output at 64 tokens per ten-second window. Adjust
`max_audio_seconds` and `max_new_tokens` as bundle variables if required.

## Updating the model

Run step 3 again to register and promote a new immutable version, then run step 4
to update the endpoint. The existing endpoint remains on its current
configuration until the replacement is ready.

## Scale-to-zero expectations

After approximately 30 traffic-free minutes, the endpoint can scale to zero.
The next request is a cold start and may take from seconds to several minutes;
there is no cold-start latency SLA or guarantee of immediate GPU capacity.
Disable scale-to-zero when consistent request latency is required.

In the September 2026 validation of this 17.6 GB packaged model, a real request
after the endpoint reported `Scaled to zero` took about 309 seconds end to end;
model inference itself took about 0.5 seconds once the replica was ready. Treat
this as an observation, not a latency guarantee.
