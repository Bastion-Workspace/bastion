-- Store per-tool-call trace for llm_agent steps (iteration, tool_name, args, result, timing).
ALTER TABLE agent_execution_steps
    ADD COLUMN IF NOT EXISTS tool_call_trace JSONB;
