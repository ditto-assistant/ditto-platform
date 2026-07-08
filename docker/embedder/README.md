# Code-embedding image (baked model)

Container image for the code-embedding anti-copy service deployed on **Cloud Run**
(scale-to-zero, authenticated). It is [text-embeddings-inference][tei] with the
model weights baked in at build time, so a cold start loads them from local disk
instead of re-downloading from the Hugging Face hub on every wake.

- `Dockerfile` — two stages: fetch the model with the HF CLI, then copy it into
  the TEI runtime and serve it from `/model` on port 8080. TEI base is pinned to
  `cpu-1.8.2` (the release that fixed the Qwen3 CPU/Intel-MKL bug — required since a
  scale-to-zero cold start can land on any CPU).
- `cloudbuild.yaml` — build + push to Artifact Registry, passing the `MODEL_ID`
  build-arg (which `gcloud builds submit --tag` cannot).

The Artifact Registry repo, the Cloud Run service, and the IAM binding that lets
the platform app service account invoke it are Terraform in the **infra** repo
(`terraform/envs/gcp-platform/embedder.tf`). The platform only owns this image and
the client (`ditto/api_server/embedding/`).

## Build

```sh
cd docker/embedder
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions=_IMAGE=<region>-docker.pkg.dev/<project>/embedder/embedder:qwen3-0.6b
```

Keep the image tag tied to the model (`qwen3-0.6b`, `jina-v2`) so a rollback is a
tag change, not a rebuild. The tag Cloud Run pulls is a Terraform var
(`embedder_image`) in the infra repo.

## Model choice

The build defaults to the primary `Qwen/Qwen3-Embedding-0.6B`. For the cheaper CPU
fallback, pass the build-args and a matching tag:

```sh
  --substitutions=_IMAGE=...:jina-v2,_MODEL_ID=jinaai/jina-embeddings-v2-base-code
```

`_MODEL_ID` must match `CODE_EMBEDDER_MODEL` in the platform env — it is the
vector's provenance tag and the gate compares only same-model vectors. Model
trade-offs are in `docs/code-embedder.md`.

## Why baked, why Cloud Run

The workload is embed-once-per-upload, latency-tolerant, low-QPS, so an
authenticated scale-to-zero Cloud Run service costs nothing at idle and needs no
vector DB. Scale-to-zero re-creates the container after idle; baking the weights
keeps that cold start to a model load (a few seconds) rather than a hub download
(tens of seconds), which the client's raised `CODE_EMBEDDER_TIMEOUT_SECONDS`
absorbs. The client authenticates with a Google-signed identity token from the
metadata server (`CODE_EMBEDDER_AUTH=gcp_id_token`); there is no static secret.

[tei]: https://github.com/huggingface/text-embeddings-inference
