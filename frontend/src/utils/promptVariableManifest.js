/**
 * Single source of truth for Agent Factory prompt variables.
 * Consumed by PromptTemplateEditor autocomplete, wire dropdown (agentFactoryTypeWiring),
 * and wiring validator. Mirrors backend RUNTIME_VARS and _is_runtime_var() in
 * llm-orchestrator/orchestrator/tools/agent_factory_tools.py
 */

export const PROMPT_VARIABLE_GROUPS = {
  conversation: 'Conversation',
  datetime: 'Date & time',
  editor: 'Editor (open document)',
  editor_refs: 'Referenced files',
  user: 'User',
  trigger: 'Trigger',
  other: 'Other context',
};

/**
 * @typedef {Object} PromptVariable
 * @property {string} key - The {key} token
 * @property {keyof PROMPT_VARIABLE_GROUPS} group
 * @property {string} label - Human-readable name
 * @property {string} description - Shown in autocomplete tooltip
 * @property {boolean} [alwaysAvailable] - Solid indicator; always filled
 * @property {boolean} [requiresOpenFile] - Dashed indicator; only when doc open
 * @property {boolean} [scheduleOnly] - Clock indicator; only for scheduled/webhook triggers
 */

/** @type {PromptVariable[]} */
export const PROMPT_VARIABLES = [
  // Conversation
  { key: 'query', group: 'conversation', label: 'Current message', description: "The user's current message or request.", alwaysAvailable: true },
  { key: 'query_length', group: 'conversation', label: 'Message length', description: 'Character count of the user message. Use {{#query_length > 200}} for long-query prompts.', alwaysAvailable: true },
  { key: 'history', group: 'conversation', label: 'Chat history', description: 'Recent conversation turns when chat history is enabled for the profile.', alwaysAvailable: false },
  { key: 'trigger_input', group: 'trigger', label: 'Trigger payload', description: 'Payload from the trigger (scheduled run or webhook).', scheduleOnly: true },

  // Date & time
  { key: 'today', group: 'datetime', label: 'Today (start)', description: "Start of today in the user's timezone (ISO 8601).", alwaysAvailable: true },
  { key: 'today_end', group: 'datetime', label: 'Today (end)', description: 'End of today (23:59:59).', alwaysAvailable: true },
  { key: 'tomorrow', group: 'datetime', label: 'Tomorrow', description: "Start of tomorrow in the user's timezone.", alwaysAvailable: true },
  { key: 'today_day_of_week', group: 'datetime', label: 'Day of week', description: 'Day name (Monday, Tuesday, etc.) in user timezone.', alwaysAvailable: true },

  // Editor (open document)
  { key: 'editor', group: 'editor', label: 'Full file content', description: 'Full content of the open file with FILE header. Empty if no file open.', requiresOpenFile: true },
  { key: 'editor_refs', group: 'editor_refs', label: 'All refs combined', description: 'All referenced file contents combined. Empty if no refs or no file open.', requiresOpenFile: true },
  { key: 'editor_document_id', group: 'editor', label: 'Document ID', description: "Document ID; use when a tool needs to save or edit the document.", requiresOpenFile: true },
  { key: 'editor_filename', group: 'editor', label: 'Filename', description: 'Base filename of the open file (e.g. chapter_01.md).', requiresOpenFile: true },
  { key: 'editor_length', group: 'editor', label: 'Character count', description: 'Character count of the open file. Use in conditionals: {{#editor_length > 5000}}...{{/editor_length > 5000}} for long docs, {{#editor_length < 5000}} for short.', requiresOpenFile: true },
  { key: 'editor_document_type', group: 'editor', label: 'Document type', description: "Frontmatter type (e.g. fiction, outline), lowercased. Use in branch conditions.", requiresOpenFile: true },
  { key: 'editor_cursor_offset', group: 'editor', label: 'Cursor offset', description: 'Character position of the cursor (-1 if not applicable).', requiresOpenFile: true },
  { key: 'editor_selection', group: 'editor', label: 'Selection', description: 'Currently selected text, if any.', requiresOpenFile: true },
  { key: 'editor_current_section', group: 'editor', label: 'Current section (follows cursor)', description: 'Content of the ## section containing the cursor. Use when you only want the section under the cursor.', requiresOpenFile: true },
  { key: 'editor_current_heading', group: 'editor', label: 'Current heading', description: 'Heading line of the section under the cursor (e.g. ## Chapter 3).', requiresOpenFile: true },
  { key: 'editor_previous_section', group: 'editor', label: 'Previous section', description: 'Content of the section immediately before the current one. Empty if cursor is in the first section.', requiresOpenFile: true },
  { key: 'editor_next_section', group: 'editor', label: 'Next section', description: 'Content of the section immediately after the current one. Empty if cursor is in the last section.', requiresOpenFile: true },
  { key: 'editor_section_index', group: 'editor', label: 'Section index', description: 'Zero-based index of the current section.', requiresOpenFile: true },
  { key: 'editor_adjacent_sections', group: 'editor', label: 'Adjacent sections (around cursor)', description: 'Previous, current, and next ## sections combined — context around the cursor without sending the whole file.', requiresOpenFile: true },
  { key: 'editor_total_sections', group: 'editor', label: 'Total sections', description: 'Total number of sections in the file.', requiresOpenFile: true },
  { key: 'editor_is_first_section', group: 'editor', label: 'Is first section', description: '"true" when cursor is in the first section; empty otherwise. Use {{#editor_is_first_section}} for start-of-doc content.', requiresOpenFile: true },
  { key: 'editor_is_last_section', group: 'editor', label: 'Is last section', description: '"true" when cursor is in the last section; empty otherwise.', requiresOpenFile: true },
  {
    key: 'editor_linked_notes',
    group: 'editor',
    label: 'Linked notes (wikilinks)',
    description:
      'Comma-separated [[Title]] wikilinks found in the open file. Empty if none. Use when you need to know what notes this document connects to.',
    requiresOpenFile: true,
  },
  { key: 'editor_ref_count', group: 'editor_refs', label: 'Ref count', description: 'Number of loaded ref_* files. Use {{#editor_ref_count > 0}} when refs exist.', requiresOpenFile: true },

  // User
  { key: 'profile', group: 'user', label: 'User profile', description: "Current user's profile (name, email, timezone, etc.) when profile loading is enabled.", alwaysAvailable: false },
  { key: 'user_weather_location', group: 'user', label: 'Weather location', description: "User's configured weather location when set.", alwaysAvailable: false },

  // Other context
  { key: 'last_tool_results', group: 'other', label: 'Last tool results', description: 'JSON of the most recent tool results from the conversation. Empty if none.', alwaysAvailable: false },
  { key: 'document_context', group: 'other', label: 'Pinned document content', description: 'Full content of the pinned document (when one is set). Empty if no pin.', alwaysAvailable: false },
  { key: 'pinned_document_id', group: 'other', label: 'Pinned document ID', description: 'Document ID of the pinned document. Empty if no pin.', alwaysAvailable: false },
  {
    key: 'current_item',
    group: 'other',
    label: 'Fan-out item',
    description: 'Current list element when llm_agent/deep_agent uses fan_out (default item_variable).',
    alwaysAvailable: false,
  },
];

const MANIFEST_KEYS = new Set(PROMPT_VARIABLES.map((v) => v.key));

/**
 * Mirrors backend _is_runtime_var(): true if ref is a known runtime var or editor_refs_* variable.
 * @param {string} ref - The placeholder name inside braces (e.g. "today", "editor_refs_rules")
 * @returns {boolean}
 */
export function isRuntimeVar(ref) {
  if (!ref || typeof ref !== 'string') return false;
  const trimmed = ref.trim();
  return MANIFEST_KEYS.has(trimmed) || trimmed.startsWith('editor_refs_');
}

/**
 * Dynamic variable patterns for editor_refs_* (from frontmatter ref_*).
 * Shown in autocomplete as fillable templates. Backend resolves at runtime from ref_* keys.
 */
export const DYNAMIC_VARIABLE_PATTERNS = [
  { key: 'editor_refs_{category}', label: 'Ref by category', description: 'Replace {category} with ref key (e.g. editor_refs_rules from ref_rules: ./rules.md).', group: 'editor_refs' },
  { key: 'editor_refs_{category}_toc', label: 'Ref TOC', description: 'Headings only for that ref file.', group: 'editor_refs' },
  { key: 'editor_refs_{category}_current', label: 'Ref section that follows cursor', description: 'Section in the ref file whose heading matches the current section in the open document. Replace {category} with ref key (e.g. editor_refs_outline_current).', group: 'editor_refs' },
  { key: 'editor_refs_{category}_adjacent', label: 'Ref sections around cursor', description: 'Previous, current, and next sections in the ref file around the match for the current heading. Use editor_refs_outline_adjacent (replace outline with your ref key) for ref content that follows the cursor.', group: 'editor_refs' },
  { key: 'editor_refs_{category}_previous', label: 'Ref previous section', description: 'Section before the cursor-matched section in that ref file. Replace {category} with ref key (e.g. editor_refs_outline_previous).', group: 'editor_refs' },
  { key: 'editor_refs_{category}_next', label: 'Ref next section', description: 'Section after the cursor-matched section in that ref file. Replace {category} with ref key (e.g. editor_refs_outline_next).', group: 'editor_refs' },
  { key: 'editor_refs_{category}_section:Name', label: 'Ref named section', description: 'Section whose heading contains Name (case-insensitive).', group: 'editor_refs' },
  { key: 'editor_refs_{prefix}*', label: 'Ref wildcard', description: 'All refs whose category starts with prefix (e.g. editor_refs_character_*).', group: 'editor_refs' },
];
