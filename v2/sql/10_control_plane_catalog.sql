CREATE SCHEMA IF NOT EXISTS workspace.ai_control_plane
COMMENT 'Training-only enterprise AI control plane';

CREATE TABLE IF NOT EXISTS workspace.ai_control_plane.model_catalog (
    catalog_entry_id STRING,
    route_name STRING,
    model_id STRING,
    provider STRING,
    capability_tier STRING,
    governance_status STRING,
    active BOOLEAN,
    allowed_for_consequential_decisions BOOLEAN,
    input_cost_usd_per_million DOUBLE,
    output_cost_usd_per_million DOUBLE,
    qualification_sample_size BIGINT,
    evidence_source STRING,
    price_source STRING,
    price_verified_date DATE,
    notes STRING,
    updated_at TIMESTAMP
)
USING DELTA
COMMENT 'Governed training inventory of AI routing options';

CREATE TABLE IF NOT EXISTS workspace.ai_control_plane.model_benchmarks (
    benchmark_id STRING,
    model_id STRING,
    route_name STRING,
    orchestration_pattern STRING,
    source_lab STRING,
    case_id STRING,
    outcome_valid BOOLEAN,
    model_calls BIGINT,
    input_tokens BIGINT,
    output_tokens BIGINT,
    latency_ms DOUBLE,
    estimated_model_cost_usd DOUBLE,
    external_action_executed BOOLEAN,
    measured_at TIMESTAMP
)
USING DELTA
COMMENT 'Measured model and orchestration benchmark results';

CREATE TABLE IF NOT EXISTS workspace.ai_control_plane.model_usage_ledger (
    event_id STRING,
    source_lab STRING,
    route_name STRING,
    model_id STRING,
    model_calls BIGINT,
    input_tokens BIGINT,
    output_tokens BIGINT,
    estimated_model_cost_usd DOUBLE,
    latency_ms DOUBLE,
    outcome_valid BOOLEAN,
    external_action_executed BOOLEAN,
    event_time TIMESTAMP
)
USING DELTA
COMMENT 'Training FinOps ledger populated by application telemetry';
