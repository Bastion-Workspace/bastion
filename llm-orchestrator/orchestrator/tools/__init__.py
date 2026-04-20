"""
Orchestrator Tools - LangGraph tools using backend gRPC services
"""

from orchestrator.tools.document_tools import (
    search_documents_tool,
    get_document_content_tool,
    find_document_by_path_tool,
    search_by_tags_tool,
    pick_random_file_tool,
    search_within_document_tool,
    get_document_metadata_tool,
    DOCUMENT_TOOLS
)
from orchestrator.tools.document_creation_tools import create_typed_document_tool
import orchestrator.tools.document_creation_tools  # noqa: F401 - register_action side effect
from orchestrator.tools.document_editing_tools import (
    list_document_proposals_tool,
    get_proposal_details_tool,
    reject_document_proposal_tool,
)
import orchestrator.tools.document_editing_tools  # noqa: F401 - register_action at load
import orchestrator.tools.help_search_tool  # noqa: F401 - register_action at load

from orchestrator.tools.web_tools import (
    search_web_tool,
    crawl_web_content_tool,
    WEB_TOOLS
)
import orchestrator.tools.browser_session_tools  # noqa: F401 - register_action at load
import orchestrator.tools.browser_automation_tools  # noqa: F401 - register_action at load

from orchestrator.tools.enhancement_tools import (
    enhance_query_tool,
    search_conversation_cache_tool,
    ENHANCEMENT_TOOLS
)

from orchestrator.tools.segment_search_tools import (
    search_segments_across_documents_tool,
    extract_relevant_content_section,
    SEGMENT_SEARCH_TOOLS
)

from orchestrator.tools.math_tools import (
    calculate_expression_tool,
    list_available_formulas_tool,
    MATH_TOOLS
)

from orchestrator.tools.math_formulas import (
    evaluate_formula_tool
)

from orchestrator.tools.unit_conversions import (
    convert_units_tool
)

from orchestrator.tools.random_tools import (
    random_number_tool
)

from orchestrator.tools.utility_tools import (
    adjust_number_tool,
    adjust_date_tool,
    parse_date_tool,
    compare_dates_tool,
    set_value_tool,
    toggle_boolean_tool,
    append_to_list_tool,
    get_list_length_tool,
)
import orchestrator.tools.agent_invocation_tools  # noqa: F401 - register_action at load
import orchestrator.tools.delegation_tools  # noqa: F401 - register_action delegate_to at load
import orchestrator.tools.agent_communication_tools  # noqa: F401 - register_action at load
import orchestrator.tools.agent_conversation_tools  # noqa: F401 - register_action at load
import orchestrator.tools.agent_workspace_tools  # noqa: F401 - register_action at load
import orchestrator.tools.consensus_tools  # noqa: F401 - register_action at load
import orchestrator.tools.agent_goal_tools  # noqa: F401 - register_action at load
import orchestrator.tools.agent_task_tools  # noqa: F401 - register_action at load
import orchestrator.tools.agent_governance_tools  # noqa: F401 - register_action at load
import orchestrator.tools.agent_memory_tools  # noqa: F401 - register_agent_memory_actions at load
import orchestrator.tools.team_tools  # noqa: F401 - register_action at load
import orchestrator.tools.local_proxy_tools  # noqa: F401 - register_action at load
import orchestrator.tools.code_workspace_tools  # noqa: F401 - register_action at load

from orchestrator.tools.visualization_tools import (
    create_chart_tool,
    VISUALIZATION_TOOLS
)

from orchestrator.tools.image_search_tools import (
    search_images_tool,
    IMAGE_SEARCH_TOOLS
)

from orchestrator.tools.image_query_analyzer import analyze_image_query

from orchestrator.tools.face_analysis_tools import (
    identify_faces_in_image,
    FACE_ANALYSIS_TOOLS
)

from orchestrator.tools.system_modeling_tools import (
    design_system_component_tool,
    simulate_system_failure_tool,
    get_system_topology_tool
)

from orchestrator.tools.data_workspace_tools import (
    list_data_workspaces_tool,
    get_workspace_schema_tool,
    resolve_workspace_link_tool,
    query_data_workspace_tool,
    create_workspace_table_tool,
    insert_workspace_rows_tool,
    update_workspace_rows_tool,
    delete_workspace_rows_tool,
    DATA_WORKSPACE_TOOLS
)

from orchestrator.tools.weather_tools import get_weather_tool
from orchestrator.tools.email_tools import (
    get_emails_tool,
    search_emails_tool,
    get_email_thread_tool,
    read_email_tool,
    move_email_tool,
    get_email_folders_tool,
    update_email_tool,
    create_draft_tool,
    get_email_statistics_tool,
    send_email_tool,
    reply_to_email_tool,
)
from orchestrator.tools.calendar_tools import (
    list_calendars_tool,
    get_calendar_events_tool,
    get_event_by_id_tool,
    create_event_tool,
    update_event_tool,
    delete_event_tool,
)
from orchestrator.tools.contact_tools import (
    get_contacts_tool,
    get_contact_by_id_tool,
    create_contact_tool,
    update_contact_tool,
    delete_contact_tool,
    search_contacts_tool,
)
import orchestrator.tools.m365_graph_todo_tools  # noqa: F401 - register_action at load
import orchestrator.tools.m365_graph_files_tools  # noqa: F401
import orchestrator.tools.m365_graph_onenote_tools  # noqa: F401
import orchestrator.tools.m365_graph_planner_tools  # noqa: F401
import orchestrator.tools.devops_tools  # noqa: F401 - register_action at load
from orchestrator.tools.account_tools import list_accounts_tool
from orchestrator.tools.navigation_tools import (
    create_location_tool,
    list_locations_tool,
    delete_location_tool,
    compute_route_tool,
    save_route_tool,
    list_saved_routes_tool,
)
from orchestrator.tools.rss_tools import (
    add_rss_feed_tool,
    list_rss_feeds_tool,
    refresh_rss_feed_tool,
    get_rss_articles_tool,
    search_rss_tool,
    list_starred_rss_articles_tool,
    delete_rss_feed_tool,
    mark_article_read_tool,
    mark_article_unread_tool,
    set_article_starred_tool,
    get_unread_counts_tool,
    toggle_feed_active_tool,
)
from orchestrator.tools.org_capture_tools import (
    add_org_inbox_item_tool,
    capture_journal_entry_tool,
)
import orchestrator.tools.org_journal_tools  # noqa: F401 - register_action at load
from orchestrator.tools.org_journal_tools import (
    get_journal_entry_tool,
    get_journal_entries_tool,
    update_journal_entry_tool,
    list_journal_entries_tool,
    search_journal_tool,
    JOURNAL_TOOLS,
)
from orchestrator.tools.todo_tools import (
    list_todos_tool,
    create_todo_tool,
    update_todo_tool,
    toggle_todo_tool,
    delete_todo_tool,
    archive_done_tool,
    refile_todo_tool,
    discover_refile_targets_tool,
)
from orchestrator.tools.org_content_tools import (
    parse_org_structure_tool,
    search_org_headings_tool,
    get_org_statistics_tool,
)
from orchestrator.tools.editor_navigation_tools import (
    editor_list_sections_tool,
    editor_get_section_tool,
    editor_get_sections_tool,
    editor_search_content_tool,
    editor_get_ref_section_tool,
)
from orchestrator.tools.image_generation_tools import (
    generate_image_tool,
)
from orchestrator.tools.text_transform_tools import (
    summarize_text_tool,
    extract_structured_data_tool,
    transform_format_tool,
    merge_texts_tool,
    compare_texts_tool,
)
from orchestrator.tools.session_memory_tools import (
    clipboard_store_tool,
    clipboard_get_tool,
)
import orchestrator.tools.artifact_tools  # noqa: F401 - register_action create_artifact
from orchestrator.tools.planning_tools import (
    create_plan_tool,
    get_plan_tool,
    update_plan_step_tool,
    add_plan_step_tool,
)
from orchestrator.tools.file_creation_tools import (
    create_user_file_tool,
    create_user_folder_tool,
    list_folders_tool,
)
import orchestrator.tools.cli_media_tools  # noqa: F401 - register_action at load
from orchestrator.tools.cli_media_tools import (
    transcode_media_tool,
    extract_audio_tool,
    trim_media_tool,
    burn_subtitles_tool,
    get_media_info_tool,
    download_media_tool,
    read_media_metadata_tool,
)
import orchestrator.tools.voice_tools  # noqa: F401 - register_action at load
from orchestrator.tools.voice_tools import transcribe_audio_tool
import orchestrator.tools.cli_document_tools  # noqa: F401 - register_action at load
from orchestrator.tools.cli_document_tools import (
    convert_document_tool,
    ocr_image_tool,
    extract_pdf_text_tool,
    split_pdf_tool,
    merge_pdfs_tool,
)
import orchestrator.tools.cli_pdf_tools  # noqa: F401 - register_action at load
from orchestrator.tools.cli_pdf_tools import (
    render_pdf_pages_tool,
    compress_pdf_tool,
    convert_pdfa_tool,
)
import orchestrator.tools.cli_image_tools  # noqa: F401 - register_action at load
from orchestrator.tools.cli_image_tools import (
    convert_image_tool,
    optimize_image_tool,
    render_diagram_tool,
)
import orchestrator.tools.cli_utility_tools  # noqa: F401 - register_action at load
from orchestrator.tools.cli_utility_tools import (
    generate_qr_code_tool,
    render_svg_tool,
)
from orchestrator.tools.notification_tools import (
    notify_user_tool,
    send_channel_message_tool,
    schedule_reminder_tool,
    NOTIFICATION_TOOLS,
)

# Load tool modules that only need to run register_action() for GetActions / LLM Agent
import orchestrator.tools.agent_factory_tools  # noqa: F401 - register_action at load
import orchestrator.tools.agent_monitoring_tools  # noqa: F401 - register_action at load
import orchestrator.tools.data_connection_tools  # noqa: F401 - register_action at load
import orchestrator.tools.control_pane_tools  # noqa: F401 - register_action at load
# Expose agent_factory tools for automation engine skill resolution (getattr(tools_module, name))
from orchestrator.tools.agent_factory_tools import (
    list_available_actions_tool,
    validate_playbook_wiring_tool,
    create_agent_profile_tool,
    create_playbook_tool,
    assign_playbook_to_agent_tool,
    create_agent_schedule_tool,
    bind_data_source_to_agent_tool,
    list_playbooks_tool,
    get_playbook_detail_tool,
    list_agent_profiles_tool,
    get_agent_profile_detail_tool,
    list_agent_schedules_tool,
    list_agent_data_sources_tool,
)
from orchestrator.tools.agent_monitoring_tools import get_agent_run_history_tool
from orchestrator.tools.data_connection_tools import (
    probe_api_endpoint_tool,
    analyze_openapi_spec_tool,
    draft_connector_definition_tool,
    validate_connector_definition_tool,
    test_connector_endpoint_tool,
    create_data_connector_tool,
    list_data_connectors_tool,
    get_data_connector_detail_tool,
    update_data_connector_tool,
    bulk_scrape_urls_tool,
    get_bulk_scrape_status_tool,
)
import orchestrator.tools.user_profile_tools  # noqa: F401 - register_action at load
import orchestrator.tools.scratchpad_tools  # noqa: F401 - register_action at load
from orchestrator.tools.user_profile_tools import (
    get_my_profile_tool,
    get_user_facts_tool,
    save_user_fact_tool,
)
from orchestrator.tools.local_proxy_tools import (
    local_screenshot_tool,
    local_clipboard_read_tool,
    local_clipboard_write_tool,
    local_system_info_tool,
    local_desktop_notify_tool,
    local_shell_execute_tool,
    local_read_file_tool,
    local_list_directory_tool,
    local_write_file_tool,
    local_list_processes_tool,
    local_open_url_tool,
    get_available_local_proxy_tools,
)
from orchestrator.tools.code_workspace_tools import (
    code_open_workspace_tool,
    code_file_tree_tool,
    code_search_files_tool,
    code_git_info_tool,
)
import orchestrator.tools.reference_file_loader  # noqa: F401 - register_action at load
import orchestrator.tools.file_analysis_tools  # noqa: F401 - register_action at load
import orchestrator.tools.file_editing_tools  # noqa: F401 - register_action at load
import orchestrator.tools.document_creation_tools  # noqa: F401 - register_action at load
import orchestrator.tools.lesson_tools  # noqa: F401 - register_action at load
import orchestrator.tools.knowledge_graph_tools  # noqa: F401 - register_action at load
import orchestrator.tools.skill_acquisition_tools  # noqa: F401 - register_action at load

# Register plugin tools (Zone 4) at startup so they appear in GetActions
try:
    from orchestrator.plugins.plugin_loader import load_plugins
    load_plugins()
except Exception:  # noqa: S110
    pass  # Plugins are optional; discovery/registration failures are non-fatal

# GitHub OAuth tools must register after plugins: GitHubPlugin used to overwrite these with
# PAT-based methods that do not accept connection_id, breaking github:{id}:github_* playbooks.
import orchestrator.tools.github_tools  # noqa: F401, E402 - register_action side effect

__all__ = [
    # Document tools
    'search_documents_tool',
    'get_document_content_tool',
    'find_document_by_path_tool',
    'search_by_tags_tool',
    'pick_random_file_tool',
    'search_within_document_tool',
    'get_document_metadata_tool',
    'create_typed_document_tool',
    'DOCUMENT_TOOLS',
    # Document edit proposal tools (list/get/reject)
    'list_document_proposals_tool',
    'get_proposal_details_tool',
    'reject_document_proposal_tool',
    # Web tools
    'search_web_tool',
    'crawl_web_content_tool',
    'WEB_TOOLS',
    # Enhancement tools
    'enhance_query_tool',
    'search_conversation_cache_tool',
    'ENHANCEMENT_TOOLS',
    # Segment search tools
    'search_segments_across_documents_tool',
    'extract_relevant_content_section',
    'SEGMENT_SEARCH_TOOLS',
    # Math tools
    'calculate_expression_tool',
    'evaluate_formula_tool',
    'convert_units_tool',
    'random_number_tool',
    'list_available_formulas_tool',
    'MATH_TOOLS',
    # Utility tools
    'adjust_number_tool',
    'adjust_date_tool',
    'parse_date_tool',
    'compare_dates_tool',
    'set_value_tool',
    'toggle_boolean_tool',
    'append_to_list_tool',
    'get_list_length_tool',
    # Visualization tools
    'create_chart_tool',
    'VISUALIZATION_TOOLS',
    # Image search tools
    'search_images_tool',
    'IMAGE_SEARCH_TOOLS',
    'analyze_image_query',
    # Face analysis tools
    'identify_faces_in_image',
    'FACE_ANALYSIS_TOOLS',
    # System modeling tools
    'design_system_component_tool',
    'simulate_system_failure_tool',
    'get_system_topology_tool',
    # Data workspace tools
    'list_data_workspaces_tool',
    'get_workspace_schema_tool',
    'resolve_workspace_link_tool',
    'query_data_workspace_tool',
    'create_workspace_table_tool',
    'insert_workspace_rows_tool',
    'update_workspace_rows_tool',
    'delete_workspace_rows_tool',
    'DATA_WORKSPACE_TOOLS',
    # Weather tools
    'get_weather_tool',
    # Email tools
    'get_emails_tool',
    'search_emails_tool',
    'get_email_thread_tool',
    'read_email_tool',
    'move_email_tool',
    'get_email_folders_tool',
    'update_email_tool',
    'create_draft_tool',
    'get_email_statistics_tool',
    'send_email_tool',
    'reply_to_email_tool',
    # Calendar tools
    'list_calendars_tool',
    'get_calendar_events_tool',
    'get_event_by_id_tool',
    'create_event_tool',
    'update_event_tool',
    'delete_event_tool',
    # Contact tools (O365)
    'get_contacts_tool',
    'get_contact_by_id_tool',
    'create_contact_tool',
    'update_contact_tool',
    'delete_contact_tool',
    'search_contacts_tool',
    'list_accounts_tool',
    # Navigation tools
    'create_location_tool',
    'list_locations_tool',
    'delete_location_tool',
    'compute_route_tool',
    'save_route_tool',
    'list_saved_routes_tool',
    # RSS tools
    'add_rss_feed_tool',
    'list_rss_feeds_tool',
    'refresh_rss_feed_tool',
    'get_rss_articles_tool',
    'search_rss_tool',
    'list_starred_rss_articles_tool',
    'delete_rss_feed_tool',
    'mark_article_read_tool',
    'mark_article_unread_tool',
    'set_article_starred_tool',
    'get_unread_counts_tool',
    'toggle_feed_active_tool',
    # Org capture tools
    'add_org_inbox_item_tool',
    'capture_journal_entry_tool',
    # Journal section-aware tools (read/update/list/search by date)
    'get_journal_entry_tool',
    'get_journal_entries_tool',
    'update_journal_entry_tool',
    'list_journal_entries_tool',
    'search_journal_tool',
    'JOURNAL_TOOLS',
    # Universal todo tools (task_management)
    'list_todos_tool',
    'create_todo_tool',
    'update_todo_tool',
    'toggle_todo_tool',
    'delete_todo_tool',
    'archive_done_tool',
    'refile_todo_tool',
    'discover_refile_targets_tool',
    # Org content tools (read-only parsing)
    'parse_org_structure_tool',
    'search_org_headings_tool',
    'get_org_statistics_tool',
    # Editor navigation tools (format-agnostic section browsing and search)
    'editor_list_sections_tool',
    'editor_get_section_tool',
    'editor_get_sections_tool',
    'editor_search_content_tool',
    'editor_get_ref_section_tool',
    # Image generation tools
    'generate_image_tool',
    # Bridging tools (plan steps)
    'summarize_text_tool',
    'extract_structured_data_tool',
    'transform_format_tool',
    'merge_texts_tool',
    'compare_texts_tool',
    'clipboard_store_tool',
    'clipboard_get_tool',
    'create_plan_tool',
    'get_plan_tool',
    'update_plan_step_tool',
    'add_plan_step_tool',
    # File creation tools
    'create_user_file_tool',
    'create_user_folder_tool',
    'list_folders_tool',
    # CLI media tools
    'transcode_media_tool',
    'extract_audio_tool',
    'trim_media_tool',
    'burn_subtitles_tool',
    'get_media_info_tool',
    'download_media_tool',
    'read_media_metadata_tool',
    'transcribe_audio_tool',
    # CLI document tools
    'convert_document_tool',
    'ocr_image_tool',
    'extract_pdf_text_tool',
    'split_pdf_tool',
    'merge_pdfs_tool',
    # CLI PDF tools
    'render_pdf_pages_tool',
    'compress_pdf_tool',
    'convert_pdfa_tool',
    # CLI image tools
    'convert_image_tool',
    'optimize_image_tool',
    'render_diagram_tool',
    # CLI utility tools
    'generate_qr_code_tool',
    'render_svg_tool',
    # Notification tools
    'notify_user_tool',
    'send_channel_message_tool',
    'schedule_reminder_tool',
    'NOTIFICATION_TOOLS',
    # Agent monitoring tools
    'get_agent_run_history_tool',
    # Data connection builder tools
    'probe_api_endpoint_tool',
    'analyze_openapi_spec_tool',
    'draft_connector_definition_tool',
    'validate_connector_definition_tool',
    'test_connector_endpoint_tool',
    'create_data_connector_tool',
    'list_data_connectors_tool',
    'get_data_connector_detail_tool',
    'update_data_connector_tool',
    'bulk_scrape_urls_tool',
    'get_bulk_scrape_status_tool',
    # Local proxy tools
    'local_screenshot_tool',
    'local_clipboard_read_tool',
    'local_clipboard_write_tool',
    'local_system_info_tool',
    'local_desktop_notify_tool',
    'local_shell_execute_tool',
    'local_read_file_tool',
    'local_list_directory_tool',
    'local_write_file_tool',
    'local_list_processes_tool',
    'local_open_url_tool',
    'get_available_local_proxy_tools',
    # Code workspace tools
    'code_open_workspace_tool',
    'code_file_tree_tool',
    'code_search_files_tool',
    'code_git_info_tool',
]

