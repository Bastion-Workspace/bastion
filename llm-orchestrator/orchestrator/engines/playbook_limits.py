"""Shared limits for Agent Factory playbook execution."""

# Parallel steps run concurrently; cap avoids runaway cost and rate limits.
MAX_PARALLEL_SUBSTEPS = 10

# Best-of-N: independent runs of the same agentic step; cap cost.
MAX_BEST_OF_N_SAMPLES = 5
