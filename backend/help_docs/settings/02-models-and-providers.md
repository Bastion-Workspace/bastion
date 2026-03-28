---
title: Models and providers
order: 2
---

# Models and providers

The **Models** tab in Settings lets you choose which models power chat, fast replies, image generation, and image analysis. You can also add **User LLM Providers** — your own API endpoints — and use them as model options across the app. This page describes each setting and how it affects agent behavior.

---

## Model roles

Settings > Models exposes one or more **roles**. Each role is a dropdown of available models. Typical roles include:

- **Main chat model** — Used for the primary chat experience: long answers, research-style replies, and agent turns that need strong reasoning. Choose a capable model (e.g. a large GPT or Claude variant) for best quality. If no model is set or the configured one is unavailable, the system may fall back to a default; the UI may show a warning when a fallback is in use.
- **Fast model** — Used when the system needs a quick, lightweight response: e.g. intent classification, short confirmations, or routing. Choose a faster, cheaper model to keep latency low. Some agent execution modes (e.g. “fast” or “direct”) use this instead of the main chat model.
- **Image generation model** — Used when an agent or tool generates images (e.g. image generation tools in playbooks). Select a model that supports image output if you use that feature.
- **Image analysis model** — Used when an agent or tool analyzes images (e.g. vision or image-understanding steps). Select a vision-capable model.

The exact labels (e.g. “Main chat model”, “Fast model”) may vary by instance. Changing a role updates which model is used the next time that role is needed (e.g. on the next chat message or tool call). You do not need to restart the app.

---

## How model selection affects agents

- **Chat** — Your **main chat model** is used for the main chat agent and for most Agent Factory agents unless they override it. **Fast model** is used for classification and fast paths when configured.
- **Agent Factory** — Custom agents can inherit the user’s model preference (from metadata) or use a profile-specific model. Subgraphs and steps that call the LLM typically use the **user_chat_model** from context when not overridden; if that is not set, the system falls back to the default or fast model. So setting **Main chat model** in Settings ensures your preferred model is used in chat and in custom agents that respect user preference.
- **Image generation / analysis** — Only used when a step or tool actually generates or analyzes images. If you do not use those features, you can leave them unset or set to any available option.

---

## User LLM Providers

**User LLM Providers** let you add your own LLM API endpoints (e.g. OpenAI-compatible APIs, or other providers). Once added, they appear in the model dropdowns for the roles above.

- **Add provider** — In the Models tab, find the **User LLM Providers** section and add a new provider. You usually specify:
  - **Name** — A label for the provider (e.g. “My OpenAI”, “Local Ollama”).
  - **Endpoint URL** — The API base URL (e.g. `https://api.openai.com/v1` or `http://localhost:11434/v1`).
  - **API key** (if required) — Stored and sent with requests. Keep this secure; only add providers you trust.
  - **Model list** (optional) — Some UIs let you restrict which models from that provider appear in the role dropdowns.

- **Use in roles** — After saving, the provider’s models show up in **Main chat model**, **Fast model**, and others. Select one like any other model.
- **Testing** — If the UI offers “Test” or “Verify”, use it to confirm the endpoint and key work before relying on them in chat or playbooks.

Provider credentials are stored per user. Admins may restrict who can add or edit User LLM Providers depending on instance configuration.

---

## Summary

- **Main chat model** and **Fast model** (and optionally **Image generation** / **Image analysis**) control which models power chat and agents. Set them in Settings > Models.
- Agents and subgraphs use the user’s chosen **chat** model from context when not overridden; **fast** model is used for classification and fast paths.
- **User LLM Providers** let you add your own API endpoints; their models then appear in the role dropdowns. Configure endpoint, key, and optional model list, then select the model in the appropriate role.

---

## Related

- **Settings overview** — All settings tabs.
- **External connections** — Email and messaging (separate from LLM providers).
- **Chat and agents overview** — How chat and agents use models.
- **Agent Factory overview** — Custom agents and playbooks.
