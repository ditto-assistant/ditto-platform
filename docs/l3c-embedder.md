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

## Deployed (GCP app VM)

The app VM already runs Pylon from this compose file; the embedder co-locates the
same way. In the host's rendered `.env` (infra repo's `platform_app` Ansible role):

```sh
DITTO_COMPOSE_SERVICES=pylon embedder     # bring the TEI service up alongside Pylon
L3C_EMBEDDER_URL=http://localhost:8080    # API and embedder share the host
L3C_EMBEDDER_MODEL=Qwen/Qwen3-Embedding-0.6B
L3C_EMBEDDER_MODEL_ID=Qwen/Qwen3-Embedding-0.6B
L3C_EMBEDDER_DIM=256
```

Provisioning that is **not** in this repo (it lives in the infra repo):

- Add `embedder` to the app VM's `DITTO_COMPOSE_SERVICES` and the `L3C_EMBEDDER_*`
  values (via Terraform vars → the `platform_app` role, or Secret Manager for any
  you prefer not to template). None are secret — the URL is loopback and the model
  ids are public.
- Size the VM for the model's RAM (see above) and allow one-time outbound HTTPS to
  `huggingface.co` for the first weight download (cached in the `embedder_hf_cache`
  volume thereafter).
- The service binds loopback only (published host port 8080 → container 80); no
  ingress/firewall change is needed since the API calls it over localhost.

An always-on CPU container is the cost/scale sweet spot for this workload (embed
once per upload, latency-tolerant). Scale-to-zero GPU (Cloud Run / Modal) is only
worth it if upload volume ever demands GPU; the client contract does not change.

## Re-embed sweep on a model change

Vectors are only comparable within one model. `code_embed_model` stamps each vector
with `model@revision`, and the gate compares only same-tag vectors. When you change
`L3C_EMBEDDER_MODEL`/`_REVISION`, re-embed the eligible ledger so old and new
vectors are comparable again — the same version-bump sweep pattern the validator
uses for `bench_version`.
