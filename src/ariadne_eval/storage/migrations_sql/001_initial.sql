-- Migration 001: initial schema for trajectories and steps.

CREATE TABLE trajectories (
    id VARCHAR PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    task VARCHAR NOT NULL,
    agent_name VARCHAR NOT NULL,
    agent_version VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    final_status VARCHAR NOT NULL,
    final_answer JSON,
    root_step_id VARCHAR,
    metadata JSON NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_trajectories_started_at ON trajectories (started_at);
CREATE INDEX idx_trajectories_agent_name ON trajectories (agent_name);
CREATE INDEX idx_trajectories_model_id  ON trajectories (model_id);
CREATE INDEX idx_trajectories_final_status ON trajectories (final_status);

CREATE TABLE steps (
    id VARCHAR PRIMARY KEY,
    trajectory_id VARCHAR NOT NULL,
    parent_step_id VARCHAR,
    step_type VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    status VARCHAR NOT NULL,
    payload JSON NOT NULL,
    error JSON,
    metadata JSON NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_steps_trajectory_id ON steps (trajectory_id);
