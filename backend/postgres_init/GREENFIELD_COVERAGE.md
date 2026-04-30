# Greenfield coverage matrix (generated)

Regenerate: `python backend/scripts/generate_greenfield_coverage.py`

## Legend

| Cat | Meaning |
|-----|---------|
| **A** | Heuristic: all `CREATE TABLE` names in this file match a `CREATE TABLE` in `01_init.sql`. |
| **B** | Pulled by `\ir` from a numbered wrapper `02`–`09`. |
| **C** | Brownfield-only or destructive — **do not** add to greenfield Docker init. |
| **D** | Gap: `CREATE TABLE` for at least one relation **not** found in `01` — needs merge or wrapper. |
| **R** | Review: no table create parsed (RLS/index/UPDATE/DROP-only, etc.). |

## Summary

- **A**: 42
- **B**: 27
- **C**: 18
- **R**: 81

## Per-file classification

| migration | cat | notes |
|---|---:|---|
| `004_add_folder_metadata.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `005_add_messaging_system.sql` | A | tables in 01: chat_messages, chat_rooms, message_reactions, room_encryption_keys, room_participants, user_presence |
| `006_add_global_folder_unique_constraint.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `006_add_teams_system.sql` | A | tables in 01: post_comments, post_reactions, team_invitations, team_members, team_posts, teams |
| `007_add_email_agent_tables.sql` | A | tables in 01: email_audit_log, email_rate_limits |
| `007_add_team_unread_tracking.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `008_add_team_folders.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `009_add_vectorization_exemption.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `009_fix_teams_rls_policies.sql` | R | no CREATE TABLE parsed — manual review |
| `010_add_hierarchical_exemption.sql` | R | no CREATE TABLE parsed — manual review |
| `010_fix_team_privacy_rls_policies.sql` | R | no CREATE TABLE parsed — manual review |
| `011_add_music_tables.sql` | A | tables in 01: music_cache, music_cache_metadata, music_service_configs |
| `012_add_service_type_to_music_configs.sql` | R | no CREATE TABLE parsed — manual review |
| `013_multi_source_media_support.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `014_add_service_type_to_cache_metadata.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `015_add_music_rls_policies.sql` | R | RLS/ALTER only — likely superseded by 01 policies |
| `016_add_folder_ownership.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `017_fix_messaging_rls_policies.sql` | R | no CREATE TABLE parsed — manual review |
| `017_fix_team_folder_delete_policy.sql` | R | no CREATE TABLE parsed — manual review |
| `018_fix_room_delete_rls_policy.sql` | R | no CREATE TABLE parsed — manual review |
| `019_fix_room_update_rls_policy.sql` | R | no CREATE TABLE parsed — manual review |
| `020_fix_conversations_update_rls_policy.sql` | R | no CREATE TABLE parsed — manual review |
| `021_fix_room_participants_insert_policy.sql` | R | no CREATE TABLE parsed — manual review |
| `022_fix_room_participants_select_policy.sql` | R | no CREATE TABLE parsed — manual review |
| `023_fix_circular_messaging_rls.sql` | R | no CREATE TABLE parsed — manual review |
| `024_break_rls_recursion.sql` | R | no CREATE TABLE parsed — manual review |
| `025_add_room_participants_update_policy.sql` | R | RLS/ALTER only — likely superseded by 01 policies |
| `026_fix_teams_select_policy_for_creation.sql` | R | no CREATE TABLE parsed — manual review |
| `027_fix_team_members_select_policy.sql` | R | no CREATE TABLE parsed — manual review |
| `028_fix_team_members_select_recursion.sql` | R | no CREATE TABLE parsed — manual review |
| `030_add_user_locations.sql` | A | tables in 01: user_locations |
| `031_add_learning_progress.sql` | B | included via \ir from 02-09 wrapper |
| `032_add_face_detection.sql` | A | tables in 01: detected_faces, known_identities |
| `033_face_encodings_to_qdrant.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `034_auto_cleanup_orphaned_identities.sql` | R | no CREATE TABLE parsed — manual review |
| `035_add_saved_routes.sql` | A | tables in 01: saved_routes |
| `036_add_object_detection.sql` | A | tables in 01: detected_objects, object_annotation_examples, user_object_annotations |
| `037_add_detected_objects_user_tag.sql` | R | no CREATE TABLE parsed — manual review |
| `038_add_detected_objects_original_class_name.sql` | R | no CREATE TABLE parsed — manual review |
| `039_add_message_attachments.sql` | B | included via \ir from 02-09 wrapper |
| `040_add_external_connections.sql` | A | tables in 01: connection_data_cache, connection_sync_state, external_connections, system_settings |
| `041_add_document_links.sql` | A | tables in 01: document_links |
| `042_add_agent_factory_tables.sql` | A | tables in 01: agent_data_sources, agent_discoveries, agent_execution_log, agent_profiles, agent_skills, custom_playbooks, data_source_connectors |
| `043_add_agent_schedules.sql` | A | tables in 01: agent_schedules |
| `044_add_agent_plugin_configs.sql` | A | tables in 01: agent_plugin_configs |
| `045_agent_schedule_delete_set_null.sql` | R | no CREATE TABLE parsed — manual review |
| `046_remove_schedule_playbook_id.sql` | R | no CREATE TABLE parsed — manual review |
| `047_add_agent_team_watches.sql` | C | listed brownfield-only filename |
| `048_add_agent_event_watches.sql` | B | included via \ir from 02-09 wrapper |
| `049_agent_factory_ux_cleanup.sql` | C | listed brownfield-only filename |
| `050_add_agent_service_bindings.sql` | C | listed brownfield-only filename |
| `051_add_chat_history_config.sql` | R | no CREATE TABLE parsed — manual review |
| `052_add_persona_enabled.sql` | R | no CREATE TABLE parsed — manual review |
| `053_add_auto_routable.sql` | R | no CREATE TABLE parsed — manual review |
| `054_add_document_edit_proposals.sql` | B | included via \ir from 02-09 wrapper |
| `055_add_user_llm_providers.sql` | B | included via \ir from 02-09 wrapper |
| `056_playbook_delete_set_null.sql` | R | no CREATE TABLE parsed — manual review |
| `058_add_user_lock.sql` | R | no CREATE TABLE parsed — manual review |
| `059_agent_profiles_handle_optional.sql` | R | no CREATE TABLE parsed — manual review |
| `060_add_include_user_context.sql` | R | no CREATE TABLE parsed — manual review |
| `061_add_include_datetime_context.sql` | R | no CREATE TABLE parsed — manual review |
| `062_add_execution_steps_and_playbook_versions.sql` | A | tables in 01: agent_execution_steps, playbook_versions |
| `063_add_tool_call_trace_to_execution_steps.sql` | R | no CREATE TABLE parsed — manual review |
| `064_add_user_facts.sql` | A | tables in 01: user_facts |
| `065_add_include_user_facts.sql` | R | no CREATE TABLE parsed — manual review |
| `066_enhance_user_facts.sql` | R | no CREATE TABLE parsed — manual review |
| `067_add_episodic_memory_and_fact_history.sql` | A | tables in 01: user_episodes, user_fact_history |
| `068_add_agent_skills.sql` | A | tables in 01: agent_skills |
| `069_add_document_versions.sql` | A | tables in 01: document_versions |
| `070_remove_profile_skill_ids.sql` | R | no CREATE TABLE parsed — manual review |
| `071_add_browser_session_states.sql` | B | included via \ir from 02-09 wrapper |
| `072_add_user_control_panes.sql` | A | tables in 01: user_control_panes |
| `073_personas_table.sql` | A | tables in 01: personas |
| `074_add_control_pane_refresh_interval.sql` | R | no CREATE TABLE parsed — manual review |
| `075_drop_entertainment_sync.sql` | C | listed brownfield-only filename |
| `076_builtin_agent_profiles.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `077_default_playbook_step_type.sql` | R | UPDATE/seed — brownfield data fix unless seed required |
| `078_add_device_tokens.sql` | A | tables in 01: device_tokens |
| `079_unlock_builtin_profiles.sql` | R | UPDATE/seed — brownfield data fix unless seed required |
| `080_add_groq_provider_type.sql` | B | included via \ir from 02-09 wrapper |
| `081_add_agent_profile_model_source.sql` | R | no CREATE TABLE parsed — manual review |
| `082_add_data_workspace_config.sql` | R | no CREATE TABLE parsed — manual review |
| `082_add_document_chunks_fulltext.sql` | A | tables in 01: document_chunks |
| `083_add_chunk_page_columns.sql` | B | included via \ir from 02-09 wrapper |
| `083_agent_budgets_and_execution_cost.sql` | A | tables in 01: agent_budgets |
| `084_agent_approval_queue.sql` | A | tables in 01: agent_approval_queue |
| `085_agent_memory.sql` | A | tables in 01: agent_memory |
| `086_agent_teams.sql` | C | listed brownfield-only filename |
| `087_agent_messages.sql` | C | listed brownfield-only filename |
| `088_agent_team_goals.sql` | C | listed brownfield-only filename |
| `089_agent_tasks.sql` | C | listed brownfield-only filename |
| `090_team_heartbeat.sql` | C | listed brownfield-only filename |
| `091_governance_types.sql` | R | no CREATE TABLE parsed — manual review |
| `092_team_budget_and_member_color.sql` | C | listed brownfield-only filename |
| `093_agent_teams_handle.sql` | C | listed brownfield-only filename |
| `094_team_workspace.sql` | C | listed brownfield-only filename |
| `096_team_tool_skills.sql` | C | listed brownfield-only filename |
| `097_agent_profile_chat_visible.sql` | R | no CREATE TABLE parsed — manual review |
| `098_drop_agent_profile_icon.sql` | R | no CREATE TABLE parsed — manual review |
| `099_repair_additional_tools_jsonb.sql` | R | UPDATE/seed — brownfield data fix unless seed required |
| `100_fix_agent_team_watches_fk.sql` | C | listed brownfield-only filename |
| `101_agent_teams_to_agent_lines.sql` | C | listed brownfield-only filename |
| `102_add_agent_profile_category.sql` | R | no CREATE TABLE parsed — manual review |
| `103_add_agent_factory_sidebar_categories.sql` | A | tables in 01: agent_factory_sidebar_categories |
| `104_agent_line_reference_config.sql` | R | no CREATE TABLE parsed — manual review |
| `105_agent_line_data_workspace_config.sql` | R | no CREATE TABLE parsed — manual review |
| `106_seed_rss_manager_agent.sql` | R | no CREATE TABLE parsed — manual review |
| `107_add_agent_profile_summarization.sql` | R | no CREATE TABLE parsed — manual review |
| `108_memory_session_summary.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `109_rss_starred_and_greader.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `110_add_user_voice_providers.sql` | A | tables in 01: user_voice_providers |
| `111_user_home_dashboards.sql` | A | tables in 01: user_home_dashboards |
| `112_message_branching.sql` | B | included via \ir from 02-09 wrapper |
| `113_builtin_playbook_remove_full_research_agent.sql` | R | UPDATE/seed — brownfield data fix unless seed required |
| `114_langgraph_checkpoint_user.sql` | R | no CREATE TABLE parsed — manual review |
| `115_user_document_pins.sql` | A | tables in 01: user_document_pins |
| `116_user_fact_themes.sql` | A | tables in 01: user_fact_themes |
| `117_drop_news_articles.sql` | C | listed brownfield-only filename |
| `118_document_sharing.sql` | B | included via \ir from 02-09 wrapper |
| `119_add_hedra_voice_provider.sql` | R | no CREATE TABLE parsed — manual review |
| `120_document_collab_state.sql` | A | tables in 01: document_collab_state |
| `121_governance_mode.sql` | R | no CREATE TABLE parsed — manual review |
| `122_code_workspaces.sql` | A | tables in 01: code_workspaces |
| `123_code_workspace_device_id.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `124_drop_legacy_github_tables.sql` | C | listed brownfield-only filename |
| `125_drop_agent_service_bindings.sql` | C | listed brownfield-only filename |
| `126_skill_connection_types.sql` | R | no CREATE TABLE parsed — manual review |
| `127_profile_allowed_connections.sql` | R | no CREATE TABLE parsed — manual review |
| `128_fix_persona_style_instructions.sql` | R | UPDATE/seed — brownfield data fix unless seed required |
| `129_oregon_trail_game_saves.sql` | A | tables in 01: oregon_trail_saves |
| `130_messaging_improvements.sql` | B | included via \ir from 02-09 wrapper |
| `131_skill_is_core.sql` | R | no CREATE TABLE parsed — manual review |
| `132_add_qdrant_point_id_to_chunks.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `133_skill_execution_events.sql` | A | tables in 01: skill_execution_events |
| `134_skill_depends_on.sql` | R | no CREATE TABLE parsed — manual review |
| `135_skill_candidate_versioning.sql` | R | no CREATE TABLE parsed — manual review |
| `136_skill_promotion_recommendations.sql` | A | tables in 01: skill_promotion_recommendations |
| `137_agent_artifact_shares.sql` | A | tables in 01: agent_artifact_shares |
| `138_saved_artifacts.sql` | A | tables in 01: saved_artifacts |
| `139_scratchpad_widget.sql` | R | no CREATE TABLE parsed — manual review |
| `140_control_pane_artifact_type.sql` | R | no CREATE TABLE parsed — manual review |
| `141_add_file_encryption.sql` | R | indexes only / no CREATE TABLE — check if merged in 01 |
| `142_agent_factory_rls.sql` | R | RLS/ALTER only — likely superseded by 01 policies |
| `143_federation_phase1.sql` | B | included via \ir from 02-09 wrapper |
| `144_federation_phase2.sql` | B | included via \ir from 02-09 wrapper |
| `145_federation_phase3.sql` | B | included via \ir from 02-09 wrapper |
| `146_federation_peers_participant_read.sql` | B | included via \ir from 02-09 wrapper |
| `147_federation_phase4.sql` | B | included via \ir from 02-09 wrapper |
| `148_document_chunk_index_state.sql` | B | included via \ir from 02-09 wrapper |
| `149_repair_stale_image_chunk_count.sql` | R | UPDATE/seed — brownfield data fix unless seed required |
| `150_image_sidecar_doc_type_and_filename.sql` | R | UPDATE/seed — brownfield data fix unless seed required |
| `151_document_processing_resilience.sql` | B | included via \ir from 02-09 wrapper |
| `152_kg_write_backlog.sql` | A | tables in 01: kg_write_backlog |
| `153_vector_embed_backlog.sql` | A | tables in 01: vector_embed_backlog |
| `154_agent_line_brief_snapshots.sql` | A | tables in 01: agent_line_brief_snapshots |
| `155_add_mcp_servers.sql` | A | tables in 01: mcp_servers |
| `156_greenfield_agent_line_watches_workspace.sql` | B | included via \ir from 02-09 wrapper |
| `157_remove_rss_and_deep_research_builtin_playbooks.sql` | B | included via \ir from 02-09 wrapper |
| `158_default_playbook_prompt_and_iterations.sql` | R | UPDATE/seed — brownfield data fix unless seed required |
| `159_fix_document_folders_global_update_rls.sql` | R | no CREATE TABLE parsed — manual review |
| `160_admin_tts_empty_means_browser.sql` | B | included via \ir from 02-09 wrapper |
| `161_add_openrouter_voice_provider.sql` | B | included via \ir from 02-09 wrapper |
| `162_add_zettelkasten_settings.sql` | B | included via \ir from 02-09 wrapper |
| `163_code_chunks.sql` | B | included via \ir from 02-09 wrapper |
| `164_user_shell_policy.sql` | B | included via \ir from 02-09 wrapper |
| `165_code_workspace_rls.sql` | B | included via \ir from 02-09 wrapper |
| `166_notification_router.sql` | A | tables in 01: mobile_push_tokens, notification_log |
| `167_device_capabilities_policy.sql` | B | included via \ir from 02-09 wrapper |

## Wrapper `\ir` targets (02-09)

- `031_add_learning_progress.sql`
- `039_add_message_attachments.sql`
- `048_add_agent_event_watches.sql`
- `054_add_document_edit_proposals.sql`
- `055_add_user_llm_providers.sql`
- `071_add_browser_session_states.sql`
- `080_add_groq_provider_type.sql`
- `083_add_chunk_page_columns.sql`
- `112_message_branching.sql`
- `118_document_sharing.sql`
- `130_messaging_improvements.sql`
- `143_federation_phase1.sql`
- `144_federation_phase2.sql`
- `145_federation_phase3.sql`
- `146_federation_peers_participant_read.sql`
- `147_federation_phase4.sql`
- `148_document_chunk_index_state.sql`
- `151_document_processing_resilience.sql`
- `156_greenfield_agent_line_watches_workspace.sql`
- `157_remove_rss_and_deep_research_builtin_playbooks.sql`
- `160_admin_tts_empty_means_browser.sql`
- `161_add_openrouter_voice_provider.sql`
- `162_add_zettelkasten_settings.sql`
- `163_code_chunks.sql`
- `164_user_shell_policy.sql`
- `165_code_workspace_rls.sql`
- `167_device_capabilities_policy.sql`
