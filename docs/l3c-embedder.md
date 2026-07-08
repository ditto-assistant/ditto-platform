# L3c code-embedding service

The L3c anti-copy signal embeds each uploaded crate through a self-hosted
[text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)
(TEI) service and compares vectors by cosine in the gate. Design context:
`SEMANTIC-CLONE-PREVENTION.md` (subnet repo). This doc covers running the service.

The platform is **disabled by default**: with `L3C_EMBEDDER_URL` unset it embeds
nothing (null column, no behavior change). Everything here is opt-in.

## Model

- Primary: `Qwen/Qwen3-Embedding-0.6B` — Apache-2.0, 32k context, Matryoshka output
  dims 32–1024. Best quality; ~2.5 GB RAM under TEI.
- CPU fallback (compose default): `jinaai/jina-embeddings-v2-base-code` — Apache-2.0,
  161M, 8192 context, 31 languages incl. Rust, trained on code↔code pairs. ~1–2 GB
  RAM, CPU-fast. Start here to validate the path cheaply.

Hosted APIs (voyage-code-3, zembed-1) are deliberately **not** used: agent crates
are private miner IP, so embedding them off-platform is unacceptable egress, and
hosted models are non-reproducible and per-call.

## Local

```sh
make embedder-up        # starts the `embedder` compose service (first boot pulls weights)
make smoke-embedder     # curl /embed with a code snippet, expect a vector
```

Then enable the signal in `.env` and restart the API:

```sh
L3C_EMBEDDER_URL=http://localhost:8080
L3C_EMBEDDER_MODEL=jinaai/jina-embeddings-v2-base-code   # must match what TEI serves
# L3C_EMBEDDER_DIM=256                                   # optional, Qwen3 only
```

`L3C_EMBEDDER_MODEL` is stored with every vector as `model@revision`; it must match
the model TEI actually serves (`L3C_EMBEDDER_MODEL_ID`), or provenance and the
gate's same-model comparison break.

## Deployed (Cloud Run, scale-to-zero)

The workload is embed-once-per-upload, latency-tolerant, and low-QPS, and the app
VM is a small `e2-medium` (4 GB) with no room to co-locate TEI comfortably. So the
deployed embedder is an **authenticated, scale-to-zero Cloud Run service**: it
costs nothing at idle and the platform reaches it over HTTPS with a Google-signed
identity token (no static secret, no ingress on the VM).

Two pieces, in two repos:

- **Image** — `docker/embedder/` here. TEI with the model baked in, built + pushed
  to Artifact Registry (`docker/embedder/cloudbuild.yaml`). Baking keeps a
  scale-to-zero cold start to a local model load rather than a hub download.
- **Service + wiring** — infra repo `terraform/modules/embedder`: the Artifact
  Registry repo, the `google_cloud_run_v2_service` (min instances 0, authenticated,
  startup-CPU-boost), and the `run.invoker` binding for the platform app service
  account. The `platform_app` role then renders these into the host `.env`:

```sh
L3C_EMBEDDER_URL=https://embedder-xxxx.a.run.app   # Cloud Run service URL (TF output)
L3C_EMBEDDER_MODEL=jinaai/jina-embeddings-v2-base-code  # must match the baked model
L3C_EMBEDDER_AUTH=gcp_id_token                     # mint a metadata-server ID token
L3C_EMBEDDER_TIMEOUT_SECONDS=30                    # absorb a cold-start model load
# L3C_EMBEDDER_DIM=256                             # Qwen3 only
```

None of these are secret — the URL is public (auth is enforced by IAM, not
obscurity) and the model ids are public. `L3C_EMBEDDER_AUTH=gcp_id_token` is what
makes the client attach the bearer token; leave it `none` only for an
unauthenticated/local TEI.

### Alternative: co-located compose

The service can instead run as the `embedder` compose service on the app VM
(alongside Pylon), reached over loopback — appropriate if the VM is sized up for
the model's RAM and you would rather avoid a second GCP service. Then:

```sh
DITTO_COMPOSE_SERVICES=pylon embedder     # bring TEI up alongside Pylon
L3C_EMBEDDER_URL=http://localhost:8080    # API and embedder share the host
L3C_EMBEDDER_MODEL=jinaai/jina-embeddings-v2-base-code
L3C_EMBEDDER_MODEL_ID=jinaai/jina-embeddings-v2-base-code
L3C_EMBEDDER_AUTH=none
```

This binds loopback only (host port 8080 → container 80), needs the VM sized for
the model's RAM, and needs one-time outbound HTTPS to `huggingface.co` for the
first weight download (cached in the `embedder_hf_cache` volume thereafter).

## Re-embed sweep on a model change

Vectors are only comparable within one model. `code_embed_model` stamps each vector
with `model@revision`, and the gate compares only same-tag vectors. When you change
`L3C_EMBEDDER_MODEL`/`_REVISION`, re-embed the eligible ledger so old and new
vectors are comparable again — the same version-bump sweep pattern the validator
uses for `bench_version`.
