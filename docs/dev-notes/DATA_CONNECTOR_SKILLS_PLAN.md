# Data Connector Skills — Future Enhancement

## Context

Data connectors and skills are complementary but currently separate primitives in Agent Factory:

- **Data connectors** are declarative — they describe an external API (endpoints, auth, parameters, response shape) and execute deterministic HTTP calls via the connections-service.
- **Skills** are procedural — they inject instructions and tool lists into LLM steps, guiding how the agent reasons and which tools it uses.

Today, when a data connector's endpoints are bound to an agent profile, they become callable tools. Step prompting within playbooks can govern their usage. This works and covers the immediate need.

The idea explored here is a future enhancement: **companion skills that encode domain expertise for using a data connector**, going beyond the mechanical "how to call" into "when and why to call, how to interpret results, and what multi-step patterns are effective."

## Current State (How It Works Today)

### Data Connector → Tool Flow

1. A data connector is defined in `data_source_connectors` (YAML/JSON definition)
2. Connector endpoints are bound to an agent profile via `agent_data_sources`
3. In playbooks, `tool` steps invoke them with `action: call_connector` or `action: connector:<id>:<endpoint>`
4. Backend loads definition + credentials, calls connections-service `ExecuteConnectorEndpoint`
5. `connector_executor.execute_endpoint()` handles HTTP, auth, pagination, response extraction
6. Returns `{records, count, formatted}`

### Governing Connector Usage Today

Step prompting in playbook `llm_task` steps already provides a way to guide the LLM on how to use connector-backed tools:

```yaml
steps:
  - name: analyze_contributions
    type: llm_task
    prompt: |
      Use the search_contributions tool to look up campaign finance data.
      Always search by committee_id when looking up PAC contributions.
      For individual donors, use contributor_name.
      Summarize the top 5 contributors by amount.
    tools:
      - connector:fec-api:search_contributions
```

This is functional and covers basic use cases. The connector is a tool, the step prompt provides guidance.

## Future Enhancement: Companion Skills

### What a Connector Skill Would Add

A skill wrapping a connector encodes **institutional knowledge** that goes beyond step prompting:

| Aspect | Step Prompting (Today) | Companion Skill (Future) |
|--------|------------------------|--------------------------|
| Scope | Single playbook step | Reusable across agents and playbooks |
| Knowledge | Ad-hoc instructions per step | Structured expertise with examples |
| Tool binding | Manual per step | Automatic via `required_tools` |
| Discovery | None — hardcoded in playbook | Searchable via `search_and_acquire_skills_tool` |
| Versioning | Embedded in playbook YAML | Independent version in `agent_skills` table |
| Sharing | Copy-paste between playbooks | Bind to any agent profile via `skill_ids` |

### Example: Weather API Connector Skill

```yaml
name: Weather Data Expert
description: How to effectively query and interpret weather data from the OpenWeather connector
category: data_connector
procedure: |
  When querying weather data:
  1. Use current_conditions for real-time weather at a location
  2. Use forecast for multi-day predictions (max 7 days)
  3. Always include temperature, humidity, and wind speed in summaries
  4. Format temperatures in both Fahrenheit and Celsius
  5. This connector does NOT support historical data — if asked, explain the limitation
  6. For location input, prefer city name + country code (e.g., "London,UK") over coordinates
  
  Common patterns:
  - Travel planning: fetch forecast for destination, summarize day-by-day
  - Outfit advice: fetch current conditions, focus on temperature and precipitation
  - Event planning: fetch forecast, highlight rain probability
required_tools:
  - connector:weather-api:current_conditions
  - connector:weather-api:forecast
inputs_schema:
  location: { type: text, description: "City or coordinates" }
outputs_schema:
  summary: { type: text, description: "Weather summary for the user" }
examples:
  - input: "What's the weather like in Tokyo?"
    expected_behavior: "Call current_conditions with location=Tokyo,JP, present temp/humidity/wind"
  - input: "Should I pack an umbrella for my trip to London next week?"
    expected_behavior: "Call forecast with location=London,UK and days=7, focus on precipitation probability"
```

### How It Would Work

1. **User creates a data connector** (e.g., a weather API)
2. **Optionally creates a companion skill** describing how to use it effectively
3. **Skill references connector endpoints** in `required_tools`
4. **Agent profile binds both** — the data source (for credentials/access) and the skill (for expertise)
5. **At runtime**, when the skill is loaded for an `llm_task` step:
   - Procedure text is injected as a system message (existing mechanism)
   - `required_tools` are merged into the step's tool set (existing mechanism)
   - The LLM gets both the tool and the expertise to use it well

### What Already Exists

The infrastructure is largely in place:

- **`agent_skills` table** — stores skill definitions with procedure, required_tools, examples
- **`agent_profiles.skill_ids`** — binds skills to agent profiles
- **`_resolve_and_inject_skills()`** — loads skills and injects procedure into LLM steps
- **`search_and_acquire_skills_tool`** — lets agents discover skills at runtime
- **Connector-generated tools** — connector endpoints already surface as callable tools

### What Would Need to Be Built

1. **UX: "Create Skill for This Connector" button** — auto-generates a skill scaffold from the connector definition (endpoints become `required_tools`, parameter descriptions seed the procedure template)
2. **Skill category: `data_connector`** — for filtering and discovery
3. **Validation** — ensure `required_tools` referencing connector endpoints resolve correctly when the skill is loaded
4. **Skill template generation** — given a connector definition, produce a reasonable starting procedure that documents each endpoint's purpose and parameters

### Priority

**Low — step prompting covers the immediate need.** This becomes valuable when:

- Users have many connectors and want reusable expertise patterns
- Multiple agents need to use the same connector with consistent behavior
- The skill marketplace / sharing feature is built out
- Agents need to dynamically discover and learn how to use connectors at runtime (via `search_and_acquire_skills_tool`)

## Decision

Park this as a future enhancement. Today's step prompting in playbooks is sufficient for governing connector tool usage. When the skill system matures and connector usage patterns become more complex, revisit this to add the companion skill scaffolding and `data_connector` skill category.
