# Technical Hyperspace Implementation Plan

**Objective:** Create a deterministic Multi-Domain Universal Simulation system ("Technical Hyperspace") capable of modeling complex causal failures across physical, logical, and biological domains.

## Phase 1: The Simulation Stronghold (Infrastructure)
**Goal:** Establish a dedicated container for high-performance simulation, isolating it from the main orchestration logic.

1.  **New Container**: `simulation-service`
    *   **Base Image**: Python 3.11-slim (optimized for math/physics libraries).
    *   **Core Libraries**: `numpy`, `scipy` (for math), `networkx` (for graph topology), `simpy` (for discrete event simulation).
    *   **Optional Heavy Artillery**: `PyBullet` or `Mujoco` (if physics fidelity requires it later).
2.  **Communication Protocol**: gRPC (matching existing architecture).
    *   Define `simulation_service.proto` with methods: `RunSimulation`, `GetComponentState`, `ApplyStressor`.
3.  **Topology Engine**:
    *   Implement a Directed Acyclic Graph (DAG) manager to handle component dependencies (`Part A` -> `Part B`).
    *   Support "State Modifiers" (e.g., `Integrity: 0.5`) on nodes.
4.  **Environment / Scenario Context**:
    *   Extend scenario model so each run can carry context: location type (maritime, airborne, terrestrial), optional coordinates or region, and domain tags. Enables "in the ocean" and "distance to assistance" to be first-class in tooling and narrative.

## Phase 2: The Agent's Command Link (Tooling)
**Goal:** Empower the `TechnicalHyperspaceAgent` to command the simulation service.

1.  **Tool Definitions**:
    *   `run_monte_carlo_simulation(scenario_config)`: Triggers the batch run.
    *   `define_initial_conditions(conditions_json)`: Sets up the "bad maintenance" state.
    *   `query_failure_path(simulation_id)`: Retrieves specific causal chains.
2.  **Orchestrator Integration**:
    *   Update `TechnicalHyperspaceAgent` to call these new tools via the `tools-service` or direct gRPC client.
    *   Ensure the agent can map Natural Language ("forgot a bolt") to JSON parameters (`{"component": "bolt", "state": "missing"}`).
3.  **Population and Biological Entities**:
    *   Support defining crew, passengers, or other populations as components with dependencies on life-support systems (HVAC, water, food, medical). Enables failure propagation to "people" and prepares for Phase 5 bio-layer.

## Phase 3: The Narrative War Room (Description & Reporting)
**Goal:** Translate raw simulation data into human-readable narratives and timelines.

1.  **The Describer Module**:
    *   A sub-routine in the agent that digests the `SimulationResult` JSON.
    *   **Prompt Engineering**: "You are an accident investigator. Describe the sequence of events in this JSON log as a narrative report."
2.  **Timeline Generation**:
    *   Agent generates a Markdown timeline:
        *   **T+00:00**: Maintenance Error (Bolt missing).
        *   **T+00:45**: Takeoff stress applied.
        *   **T+01:20**: Jack-screw failure.
3.  **Visualization**:
    *   Use the existing Mermaid diagramming capabilities to draw the failure tree dynamically based on the specific simulation run.
4.  **Human Impact in Narratives**:
    *   Describer and timeline generation include human/biological impact: when life-support systems fail, report "passenger welfare degraded" or "crew stress elevated" with optional time bands (e.g. critical after N hours).
5.  **Environmental Context in Narratives**:
    *   When scenario context includes location (e.g. "at sea", "50 nm from port"), narratives and timelines mention it (e.g. "Vessel adrift; nearest assistance 12 hours away").

## Phase 4: Advanced Fidelity (The "Regular Guy" Interface)
**Goal:** Allow non-technical users to input complex scenarios.

1.  **Scenario Templates**:
    *   Pre-defined "battle plans" (e.g., "Maintenance Failure", "Weather Stress", "Cyber Attack").
    *   User just fills in the blanks: "Who failed?" -> "Technician". "What part?" -> "Jack-Screw".
2.  **Knowledge Graph Integration**:
    *   Link to `vector-service` to understand what a "Jack-Screw" actually *is* so the agent knows reasonable failure modes without explicit user input.
3.  **Scenario Templates — Geography**:
    *   Templates include "Where?" (e.g. "In the ocean", "Distance to port", "Weather", "Nearest help ETA") so users describe the environment without low-level parameters.
4.  **Scenario Templates — Who Is Affected**:
    *   Templates include "Who is affected?" (e.g. crew count, passenger count, vulnerable populations) and optional "How are people modeled?" (simple dependency vs. stress/health) for future fidelity.