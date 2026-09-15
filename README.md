# Governed Agentic AI Underwriting Control Plane

[![Offline control-plane tests](https://github.com/anirudh2272/governed-agentic-ai-underwriting/actions/workflows/offline-tests.yml/badge.svg)](https://github.com/anirudh2272/governed-agentic-ai-underwriting/actions/workflows/offline-tests.yml)

A synthetic training project demonstrating how to build and
evaluate a governed AI-agent workflow with deterministic routing,
dynamic model selection, controlled MCP tools, cost limits,
human authorization, telemetry, and Databricks audit persistence.

This repository is an educational system. It does not make real
insurance decisions, bind coverage, set premiums, or grant an AI
model consequential authority.

## What the project demonstrates

- Deterministic `NO_LLM` routing when an authoritative answer exists
- Explicit-context construction and context-reduction measurement
- Governed conversational and premium-reasoning routes
- Databricks-backed model selection with fail-closed qualification
- Strict read-only tool allowlists and argument validation
- MCP tool discovery and controlled execution
- Human-review and human-approval boundaries
- Shared authoritative case state with atomic transitions
- Idempotent human-decision recording
- Token, latency, and estimated-cost telemetry
- Databricks model catalog, benchmark, usage, and decision auditing
- Single-agent versus multi-agent economic comparison

## Architecture

| Layer | Responsibility |
|---|---|
| Router | Selects deterministic, conversational, or premium reasoning |
| Policy | Enforces data, tool, step, cost, and authorization limits |
| Model selector | Chooses only active, qualified catalog entries |
| Route executor | Executes deterministic or approved model routes |
| Tool gateway | Allows only approved tools with strict arguments |
| MCP server | Exposes the governed read-only underwriting tools |
| Case-state service | Maintains atomic evidence and decision history |
| Telemetry | Records tokens, latency, routing, cost, and safety fields |
| Databricks | Stores catalog, benchmark, usage, and audit records |

## Project layout

```text
data/                 Synthetic authoritative underwriting state
labs/                 V1 foundational labs
mcp_server/           V1 MCP server
tools/                Synthetic tools and authorization controls
v2/control_plane/     Governed routing, policies, and model selection
v2/labs/              V2 labs and capstone validation programs
v2/mcp_server/        Governed V2 MCP server
v2/services/          Context, execution, state, and telemetry services
v2/sql/               Databricks control-plane table definitions
v2/data/              Measured training and capstone artifacts
```

## Validated environment

- Python 3.13.7
- Anthropic SDK 1.5.0
- MCP 2.2.0
- python-dotenv 1.2.3
- Databricks SQL Connector 4.5.0
- Databricks OAuth user-to-machine authentication

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Add your own credentials and workspace configuration to `.env`.
Never commit that file.

Required variables:

```text
ANTHROPIC_API_KEY
CLAUDE_MODEL
DATABRICKS_SERVER_HOSTNAME
DATABRICKS_HTTP_PATH
DATABRICKS_AUTH_TYPE
```

For the validated OAuth configuration:

```text
DATABRICKS_AUTH_TYPE=databricks-oauth
```

## Running the labs

The labs are designed to be completed sequentially because later
checkpoints consume artifacts produced by earlier checkpoints.

Examples:

```bash
python -m labs.lab00_setup
python -m v2.labs.lab00_environment
python -m v2.labs.lab11_control_plane
python -m v2.labs.capstone_final_audit
```

The Databricks and Anthropic labs may require configured credentials,
network access, and interactive OAuth authentication.

## Verified capstone outcome

The final synthetic capstone audit verified all 12 stage checks.

State history:

1. Evidence incomplete; policy step `REQUEST_EVIDENCE`
2. Evidence complete and verified; policy step `HUMAN_REVIEW`
3. Explicit human decision `REFER`
4. Workflow status `REFERRED_FOR_SPECIALIST_REVIEW`

The demonstrated workflow recorded no external underwriting action
and no coverage decision.

Phase B measured results:

| Metric | Measured value |
|---|---:|
| Model calls | 3 |
| MCP tool calls | 5 |
| Input tokens | 5,560 |
| Output tokens | 662 |
| Model latency | 7,923.31 ms |
| MCP latency | 13.15 ms |
| Estimated model cost | $0.00887 |
| Enforced cost limit | $0.01 |

These measurements describe one synthetic training run. They are not
production performance, pricing, or qualification claims.

## Safety and scope

- All case data is synthetic.
- Model qualification is scoped to training only.
- The model is not authorized for consequential decisions.
- Missing loss history is never interpreted as zero losses.
- Human review and approval remain application-controlled.
- Secrets belong only in the ignored `.env` file.
- Databricks writes are limited to training governance and audit data.
