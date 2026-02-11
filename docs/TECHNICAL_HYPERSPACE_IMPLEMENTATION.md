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

## Phase 5: Standards & Interoperability (The "Grand Strategy")
**Goal:** Align the simulation with global engineering and defense standards to ensure robustness and universality.

1.  **The Environmental Layer (The Map)**:
    *   **DGIF (Defence Geospatial Information Framework)**: Use for military-grade terrain, obstacle, and mission data definitions. Ideal for "Plane in White-out" scenarios where terrain data is critical.
    *   **S-100**: Use for maritime and hydrographic domains (e.g., submarines, shipping).
    *   **Scenario geography (first-class)**: Position, distance-to-help, weather, currents as scenario parameters. Use S-100 for maritime where applicable so "what happens in the ocean" and ETA to assistance drive simulation and narrative.
2.  **The Aeronautical Data Layer (The Sky)**:
    *   **AIXM (Aeronautical Information Exchange Model)**: Use for defining the "Invisible Infrastructure" of the sky.
        *   **Dynamic Data**: NOTAMs (Notice to Airmen), temporary flight restrictions, runway closures.
        *   **Static Data**: ILS Beacons, Glide Slopes, Airspace Boundaries.
        *   *Relevance:* In the "Wrong ILS Beam" scenario, AIXM defines the properties of the "Trap" beam vs the "Correct" beam.
3.  **The Asset Layer (The Machine)**:
    *   **DTDL (Digital Twin Definition Language)**: Use for defining component hierarchies (`Pilot` controls `Plane`) and properties (`Bolt_Count`).
    *   **Asset Administration Shell (AAS)**: Use for distinguishing between the "Type" (Blueprint) and "Instance" (Specific Plane with maintenance history).
4.  **The Bio-Layer (The Living)**:
    *   **ODD Protocol (Overview, Design concepts, Details)**: Standard for Agent-Based Models (ABMs) to describe crowd dynamics ("Town Square") and individual behavior rules.
    *   **Movebank Format**: Standard for tracking animal/biological movement vectors (e.g., bird flocks, wandering crowds).
    *   **SBML (Systems Biology Markup Language)**: For internal biological processes (viral replication, drug kinetics).
    *   **HL7 FHIR (Physiology Resource)**: For human health states (Pilot Heart Condition, Stress Levels).
    *   **Human factors in scenarios**: First-class modeling of crew/passenger state (e.g. stress, health, capacity) as state modifiers or sub-models. Use HL7 FHIR and ODD/crowd rules where applicable so "what happens to people" is more than a single failure node.
5.  **The Logic Layer (The Rules)**:
    *   **Technical Hyperspace Agent**: Acts as the translator between these rigorous standards and the user's natural language.

## Phase 6: The "Strategic Inspector" (Gap Analysis)
**Goal:** Enable the agent to analyze the current model and identify missing critical interactions.

1.  **The Ontology Scanner**:
    *   The agent scans the active DTDL/AAS models (`RailGun`, `ShipHull`).
    *   It queries the Knowledge Graph for "known interactions" between these types.
    *   *Example Logic:* "I see a `RailGun` (High Kinetic Energy) and a `ShipHull` (Floating Rigid Body). I do NOT see a `Recoil_Transfer` link defined. **WARNING: Physics Interaction Missing.**"
2.  **The "Missing Link" Report**:
    *   The agent proactively suggests: "You are simulating a Rail Gun firing, but the `Ship_Stability` module is not connected to the `Gun_Recoil` output. The ship will not rock. Shall I add a rigid-body force connection?"
3.  **Cross-Domain Audit**:
    *   Checks for "Orphaned" domains.
    *   *Example:* "You have a `Pilot` (Bio-Layer) and a `Cockpit` (Asset-Layer), but no `Interface` defined. How does the pilot fly the plane?"

## Phase 7: Reproducibility, UQ, and Validation (Lab-Grade / Data-Engineer Bar)
**Goal:** Make every run reproducible, traceable, and auditable; add uncertainty quantification and validation so the system meets laboratory and advanced data-engineering standards.

1.  **Reproducibility and Provenance**:
    *   **Versioned artifacts**: Topology, component definitions, propagation rules, and scenario configs are versioned (e.g. catalog or git-friendly). Every run records exactly which versions were used.
    *   **Determinism**: Given topology version + initial conditions + optional random seed, re-running yields the same result.
    *   **Provenance graph**: Each run's lineage is queryable (e.g. "this failure chain came from topology v2.1, scenario X, seed 42") for audit and replication.
2.  **Uncertainty Quantification (UQ) and Sensitivity**:
    *   **Uncertain inputs**: Represent uncertain initial conditions (e.g. fuel state, component health) as distributions; runs produce distributions of outcomes, not just point estimates.
    *   **Sensitivity analysis**: Identify which assumptions or parameters drive outcomes (e.g. Sobol indices, one-at-a-time). Narratives and timelines carry uncertainty (e.g. "T+2h–4h estimated") so decisions are not over-confident.
3.  **Validation and Calibration**:
    *   **Incident alignment**: Compare simulation outputs to real events (NTSB, MAIB, accident DBs) to check whether failure cascades match known cases.
    *   **Calibration loop**: Use observed failure rates, repair times, or outcomes to tune propagation rules or parameters so the model improves with data.
    *   **Ground-truth tests**: Regression suite of "golden" scenarios with expected failure chains; CI runs them so changes do not silently break behavior.
4.  **Schema and Model Evolution**:
    *   **Topology as data**: Topology and rules are first-class, versioned artifacts (e.g. YAML/JSON + schema). Diffs between versions; no hidden state.
    *   **Backward compatibility**: Old runs remain interpretable after schema/model changes; migration or compatibility layer so history does not rot.
    *   **Data-engineer-friendly**: Export/import in open formats (e.g. Parquet for run history, OTLP for traces) so the system fits into data lakes and pipelines.
5.  **Observability and Debuggability**:
    *   **Full causal trace**: For any failed component, query "why?" and get the path (which antecedent failed, which rule fired, at what logical time).
    *   **Step-through / replay**: Inspect a run event-by-event; optional "time travel" for debugging.
    *   **Structured logs and traces**: Runs emit structured events (e.g. OpenTelemetry-compatible) for analysis in standard observability tools.
6.  **Controlled Experimentation**:
    *   **Scenario comparison**: "Same topology, scenario A vs B" with clear diff of outcomes (e.g. side-by-side timelines, delta in failure sets).
    *   **Parameter sweeps**: Systematic variation of one or a few inputs (e.g. distance to port, crew size) with results in a table or dataset for analysis.
7.  **Standards-First and Auditable**:
    *   **Open formats**: Publish/consume standard ontologies (DTDL, S-100, FHIR, etc.) so the system plugs into existing ecosystems.
    *   **Documented assumptions**: Every propagation rule and default is documented and referenceable; "method" section is first-class so results are peer-reviewable.
8.  **Multi-Fidelity Workflow**:
    *   **Fast vs high-fidelity**: Same scenario runnable in "fast" mode (simplified rules, fewer components) for iteration and in "high-fidelity" mode for validation. Same semantics, different resolution.
    *   **Consistency checks**: When both exist, coarse and fine runs agree on high-level outcomes where they overlap.
