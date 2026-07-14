# Screening protocol dependency

The API owns queue state, leases, verdict acceptance, screening history, and
public status projection. The private `ditto-screener` repository owns the
build/run worker. They share only `ditto-screening-protocol`, pinned in
`pyproject.toml` and `uv.lock` to an exact private-repository commit.

Required credentials:

- `DITTO_SCREENER_PROTOCOL_READ_KEY`: a GitHub repository secret containing a
  read-only deploy key for `ditto-assistant/ditto-screener`. CI uses it only for
  the `uv sync` step and removes the temporary key file afterward.
- `DITTO_SCREENER_PROTOCOL_HOST_KEY`: a separate read-only deploy key installed
  in the platform VM deploy user's SSH configuration. This is a host credential,
  not a GitHub secret. It lets `scripts/update.sh` resolve the frozen protocol
  dependency before migration or process reload.

The private keys must never be committed or printed. Register only their public
halves as read-only deploy keys on `ditto-screener`.

The protocol package contains request/response models, `AgentStatus`, artifact
metadata, `SCREENING_POLICY_VERSION`, and the canonical signing function. The
API never imports worker application code.
