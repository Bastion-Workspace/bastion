"""
Orchestrator Agents - LangGraph agents using gRPC backend tools
"""

from orchestrator.agents.full_research_agent import FullResearchAgent, get_full_research_agent
from orchestrator.agents.chat_agent import ChatAgent
from orchestrator.agents.dictionary_agent import DictionaryAgent
from orchestrator.agents.data_formatting_agent import DataFormattingAgent
from orchestrator.agents.help_agent import HelpAgent
from orchestrator.agents.weather_agent import WeatherAgent, get_weather_agent
from orchestrator.agents.image_generation_agent import ImageGenerationAgent, get_image_generation_agent
# FactCheckingAgent removed - not actively used
from orchestrator.agents.rss_agent import RSSAgent, get_rss_agent
from orchestrator.agents.org_inbox_agent import OrgInboxAgent, get_org_inbox_agent
from orchestrator.agents.substack_agent import SubstackAgent, get_substack_agent
from orchestrator.agents.podcast_script_agent import PodcastScriptAgent, get_podcast_script_agent
from orchestrator.agents.org_project_agent import OrgProjectAgent, get_org_project_agent
from orchestrator.agents.entertainment_agent import EntertainmentAgent, get_entertainment_agent
from orchestrator.agents.website_crawler_agent import WebsiteCrawlerAgent, get_website_crawler_agent
from orchestrator.agents.electronics_agent import ElectronicsAgent, get_electronics_agent
from orchestrator.agents.character_development_agent import CharacterDevelopmentAgent, get_character_development_agent
from orchestrator.agents.content_analysis_agent import ContentAnalysisAgent, get_content_analysis_agent
from orchestrator.agents.fiction_editing_agent import FictionEditingAgent, get_fiction_editing_agent
from orchestrator.agents.outline_editing_agent import OutlineEditingAgent, get_outline_editing_agent
from orchestrator.agents.story_analysis_agent import StoryAnalysisAgent, get_story_analysis_agent
from orchestrator.agents.site_crawl_agent import SiteCrawlAgent, get_site_crawl_agent
from orchestrator.agents.rules_editing_agent import RulesEditingAgent, get_rules_editing_agent
from orchestrator.agents.style_editing_agent import StyleEditingAgent, get_style_editing_agent
from orchestrator.agents.proofreading_agent import ProofreadingAgent, get_proofreading_agent
from orchestrator.agents.general_project_agent import GeneralProjectAgent, get_general_project_agent
from orchestrator.agents.reference_agent import ReferenceAgent, get_reference_agent
from orchestrator.agents.knowledge_builder_agent import KnowledgeBuilderAgent, get_knowledge_builder_agent

__all__ = [
    'FullResearchAgent',
    'get_full_research_agent',
    'ChatAgent',
    'DictionaryAgent',
    'DataFormattingAgent',
    'HelpAgent',
    'WeatherAgent',
    'get_weather_agent',
    'ImageGenerationAgent',
    'get_image_generation_agent',
    # FactCheckingAgent removed - not actively used
    'RSSAgent',
    'get_rss_agent',
    'OrgInboxAgent',
    'get_org_inbox_agent',
    'SubstackAgent',
    'get_substack_agent',
    'PodcastScriptAgent',
    'get_podcast_script_agent',
    'OrgProjectAgent',
    'get_org_project_agent',
    'EntertainmentAgent',
    'get_entertainment_agent',
    'WebsiteCrawlerAgent',
    'get_website_crawler_agent',
    'ElectronicsAgent',
    'get_electronics_agent',
    'CharacterDevelopmentAgent',
    'get_character_development_agent',
    'ContentAnalysisAgent',
    'get_content_analysis_agent',
    'FictionEditingAgent',
    'get_fiction_editing_agent',
    'OutlineEditingAgent',
    'get_outline_editing_agent',
    'StoryAnalysisAgent',
    'get_story_analysis_agent',
    'SiteCrawlAgent',
    'get_site_crawl_agent',
    'RulesEditingAgent',
    'get_rules_editing_agent',
    'StyleEditingAgent',
    'get_style_editing_agent',
    'ProofreadingAgent',
    'get_proofreading_agent',
    'GeneralProjectAgent',
    'get_general_project_agent',
    'ReferenceAgent',
    'get_reference_agent',
    'KnowledgeBuilderAgent',
    'get_knowledge_builder_agent'
]

