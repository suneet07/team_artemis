# Mocks

Fixtures satisfy the contract in `../contracts/openapi.yaml` exactly. Build the whole app against
them; `VITE_USE_MOCKS=0` switches to the live API.

| Fixture | Use |
|---|---|
| `bundle_crossmodal_ready.json` | prepared optical+SAR bundle, `supported_tasks` populated |
| `query_crossmodal_success.json` | full `QueryResult` incl. evidence assets + trace |
| `trace_crossmodal_success.json` | schema-v2 trace, two tools, disagreement verdict |
| `trace_refusal_missing_input.json` | change question on a single scene → refusal card |
| `trace_param_rejected.json` | `parameter_check.passed:false` → red trace step |
| `scene_no_crs.json` | benchmark PNG, `crs_valid:false` → non-geo viewer path |
| `scene_pan_only.json` | pan-only optical, no computable indices |
| `tools.json` `tasks.json` `disagreement_causes.json` `health.json` | `/meta/*` responses |

CI check to add: validate every `trace_*.json` against `configs/trace_schema.json` with Ajv. If the
backend bumps the schema, these fail first — that is the alarm you want.
