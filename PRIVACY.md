# Privacy

The plugin sends exactly one extra HTTP header, `X-Agent-Hint`. Its payload is
a small JSON object encoded as base64url. It contains:

| Field | Content | Personal data? |
| --- | --- | --- |
| `session.session_id` | opaque identifier: the AgentScope `AgentState.session_id`, or an id supplied by the host / generated randomly | no, unless the host puts identifying text in it |
| `session.parent_session_id` | optional, only when the host configures it | same caveat |
| `session.call_index`, `session.epoch`, `session.state` | integers / enum | no |
| `priority_control` | two enum/integer values | no |
| `cache_control` | type, TTL seconds, message count | no |
| `context_management` | verb and edit type (manage requests only) | no |

The plugin does **not** read, copy, log or transmit prompts, model output, tool
arguments, file contents, credentials or user identifiers. It does not add any
field to the request body and does not persist conversation data.

The header is sent to the same model endpoint (or the explicitly configured
manage endpoint) that the harness already uses, over the same TLS connection.
Any logging done by that endpoint is governed by the endpoint operator's
policy.

`ManageClient.extra_headers` is caller-supplied; do not put secrets in the
hint itself. The hint is limited to the fields above by the protocol schema and
`validate()` rejects unknown keys.

Session ids should be opaque. If your application needs traceability, use a
random id and keep the mapping in your own storage rather than embedding user
identifiers in `session_id`.
