"""
Report Template Models - Roosevelt's "Structured Intelligence" Architecture
Pydantic models for user-defined report templates and templated research outputs
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Literal, Any, Union
from enum import Enum
from datetime import datetime
import uuid


# ===== ENUMS AND TYPES =====

class FieldType(str, Enum):
    """Types of fields in report templates"""
    TEXT = "text"                    # Free-form text
    LONG_TEXT = "long_text"         # Multi-paragraph text
    LIST = "list"                   # Bulleted/numbered list
    DATE = "date"                   # Date field
    NUMBER = "number"               # Numeric value
    URL = "url"                     # Web URL
    EMAIL = "email"                 # Email address
    PHONE = "phone"                 # Phone number
    ADDRESS = "address"             # Physical address
    IMAGE = "image"                 # Image/photo placeholder
    STRUCTURED_DATA = "structured_data"  # Key-value pairs


class RequestType(str, Enum):
    """Types of research requests"""
    GENERAL_RESEARCH = "general_research"
    TEMPLATED_REPORT = "templated_report"
    AMBIGUOUS = "ambiguous"


class TemplateScope(str, Enum):
    """Template visibility scope"""
    PRIVATE = "private"             # User's private template
    SHARED = "shared"               # Shared with specific users
    PUBLIC = "public"               # Available to all users
    SYSTEM = "system"               # Built-in system templates


# ===== TEMPLATE STRUCTURE MODELS =====

class ReportTemplateField(BaseModel):
    """Individual field within a template section"""
    field_id: str = Field(description="Unique identifier for field")
    field_name: str = Field(description="Display name of field")
    field_type: FieldType = Field(description="Type of data expected")
    description: str = Field(description="What information should go in this field")
    required: bool = Field(default=False, description="Whether field is required")
    placeholder: Optional[str] = Field(default=None, description="Example/placeholder text")
    max_length: Optional[int] = Field(default=None, description="Maximum character length")
    validation_pattern: Optional[str] = Field(default=None, description="Regex validation pattern")
    default_value: Optional[str] = Field(default=None, description="Default field value")
    order_index: int = Field(description="Display order within section")

    @validator('field_id')
    def validate_field_id(cls, v):
        """Ensure field_id is valid identifier"""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('field_id must be alphanumeric with underscores/hyphens only')
        return v


class ReportTemplateSection(BaseModel):
    """Section within a report template"""
    section_id: str = Field(description="Unique identifier for section")
    section_name: str = Field(description="Display name of section") 
    description: str = Field(description="What information should be included in this section")
    fields: List[ReportTemplateField] = Field(description="Fields within this section")
    required: bool = Field(default=False, description="Whether section is required")
    order_index: int = Field(description="Display order within template")
    collapsible: bool = Field(default=True, description="Whether section can be collapsed in UI")
    instructions: Optional[str] = Field(default=None, description="Special instructions for research agent")

    @validator('section_id')
    def validate_section_id(cls, v):
        """Ensure section_id is valid identifier"""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('section_id must be alphanumeric with underscores/hyphens only')
        return v

    @validator('fields')
    def validate_unique_field_ids(cls, v):
        """Ensure field IDs are unique within section"""
        field_ids = [field.field_id for field in v]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError('Field IDs must be unique within section')
        return v


class ReportTemplate(BaseModel):
    """Complete report template definition"""
    template_id: str = Field(description="Unique template identifier")
    template_name: str = Field(description="Human-readable template name")
    description: str = Field(description="What this template is used for")
    category: str = Field(default="general", description="Template category/group")
    sections: List[ReportTemplateSection] = Field(description="Ordered list of report sections")
    keywords: List[str] = Field(default_factory=list, description="Keywords for auto-detection")
    scope: TemplateScope = Field(default=TemplateScope.PRIVATE, description="Template visibility")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional template metadata")
    created_by: str = Field(description="User ID who created template")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Template creation timestamp")
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Last update timestamp")
    version: int = Field(default=1, description="Template version number")
    is_active: bool = Field(default=True, description="Whether template is active")

    @validator('template_id')
    def validate_template_id(cls, v):
        """Ensure template_id is valid identifier"""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('template_id must be alphanumeric with underscores/hyphens only')
        return v

    @validator('sections')
    def validate_unique_section_ids(cls, v):
        """Ensure section IDs are unique within template"""
        section_ids = [section.section_id for section in v]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError('Section IDs must be unique within template')
        return v

    @validator('keywords')
    def normalize_keywords(cls, v):
        """Normalize keywords to lowercase"""
        return [keyword.lower().strip() for keyword in v if keyword.strip()]


# ===== RESEARCH REQUEST AND ANALYSIS MODELS =====

class RequestAnalysisResult(BaseModel):
    """LLM's structured analysis of research request"""
    request_type: RequestType = Field(description="Type of research request detected")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in request type classification")
    suggested_template_id: Optional[str] = Field(default=None, description="Recommended template ID")
    template_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in template suggestion")
    reasoning: str = Field(description="Explanation for classification decision")
    detected_keywords: List[str] = Field(default_factory=list, description="Keywords that influenced decision")
    subject_entity: Optional[str] = Field(default=None, description="Main subject of research (person, company, etc.)")
    alternative_templates: List[str] = Field(default_factory=list, description="Other possible template matches")


class TemplatedResearchRequest(BaseModel):
    """Structured request for templated research"""
    query: str = Field(description="Original research query")
    template_id: str = Field(description="Template to use for research")
    custom_instructions: Optional[str] = Field(default=None, description="Additional instructions")
    priority_sections: Optional[List[str]] = Field(default=None, description="Sections to prioritize")
    research_depth: Literal["shallow", "standard", "deep"] = Field(default="standard", description="Research depth level")
    include_sources: bool = Field(default=True, description="Whether to include detailed sources")
    max_processing_time: Optional[int] = Field(default=None, description="Maximum processing time in seconds")


# ===== TEMPLATED RESEARCH OUTPUT MODELS =====

class FilledTemplateField(BaseModel):
    """Field with research content filled in"""
    field_id: str = Field(description="Field identifier")
    field_name: str = Field(description="Field display name")
    field_type: FieldType = Field(description="Field data type")
    content: Optional[Union[str, List[str], Dict[str, Any], int, float]] = Field(default=None, description="Research content for field")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in field content")
    sources: List[str] = Field(default_factory=list, description="Source IDs that contributed to field")
    is_filled: bool = Field(description="Whether field has content")
    research_notes: Optional[str] = Field(default=None, description="Agent notes about field research")


class FilledTemplateSection(BaseModel):
    """Section with research content filled in"""
    section_id: str = Field(description="Section identifier")
    section_name: str = Field(description="Section display name")
    fields: List[FilledTemplateField] = Field(description="Fields within section")
    completion_percentage: float = Field(ge=0.0, le=100.0, description="Percentage of section completed")
    section_confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in section")
    section_summary: Optional[str] = Field(default=None, description="Summary of section findings")
    research_challenges: List[str] = Field(default_factory=list, description="Challenges encountered researching section")


class TemplatedResearchResult(BaseModel):
    """Structured output for template-based research"""
    template_id: str = Field(description="Template used for research")
    template_name: str = Field(description="Template display name")
    original_query: str = Field(description="Original research query")
    filled_sections: List[FilledTemplateSection] = Field(description="Sections with research content")
    research_summary: str = Field(description="Overall research summary")
    total_completion_percentage: float = Field(ge=0.0, le=100.0, description="Overall template completion")
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in research")
    unfilled_sections: List[str] = Field(default_factory=list, description="Section IDs that couldn't be filled")
    unfilled_fields: List[str] = Field(default_factory=list, description="Field IDs that couldn't be filled")
    research_limitations: List[str] = Field(default_factory=list, description="Limitations encountered")
    suggested_next_steps: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    sources_used: List[Dict[str, Any]] = Field(default_factory=list, description="All sources used in research")
    processing_time: float = Field(description="Total research processing time")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Research completion timestamp")


class TemplateConfirmationRequest(BaseModel):
    """Request for user confirmation of template usage"""
    original_query: str = Field(description="Original research query")
    suggested_template: ReportTemplate = Field(description="Suggested template details")
    analysis_result: RequestAnalysisResult = Field(description="Analysis that led to suggestion")
    confirmation_message: str = Field(description="Message asking for user confirmation")
    options: List[str] = Field(description="Available user response options")


# ===== UTILITY FUNCTIONS =====

def generate_template_id(name: str) -> str:
    """Generate template ID from name"""
    # Convert to snake_case and add UUID suffix for uniqueness
    import re
    clean_name = re.sub(r'[^\w\s-]', '', name.lower())
    snake_case = re.sub(r'[-\s]+', '_', clean_name)
    return f"{snake_case}_{uuid.uuid4().hex[:8]}"


def generate_section_id(name: str) -> str:
    """Generate section ID from name"""
    import re
    clean_name = re.sub(r'[^\w\s-]', '', name.lower())
    return re.sub(r'[-\s]+', '_', clean_name)


def generate_field_id(name: str) -> str:
    """Generate field ID from name"""
    import re
    clean_name = re.sub(r'[^\w\s-]', '', name.lower())
    return re.sub(r'[-\s]+', '_', clean_name)


def create_poi_template() -> ReportTemplate:
    """Create the Person of Interest template from user example"""
    return ReportTemplate(
        template_id="person_of_interest_report",
        template_name="Professional Biography and Background Analysis",
        description="Comprehensive public information profile for business intelligence, due diligence, and professional assessment using OSINT methodologies",
        category="business_intelligence",
        keywords=["biography", "profile", "due diligence", "business intelligence", "professional background", "public records", "OSINT"],
        scope=TemplateScope.SYSTEM,
        sections=[
            ReportTemplateSection(
                section_id="poi_details",
                section_name="POI Details",
                description="Basic identifying information about the person",
                order_index=1,
                required=True,
                fields=[
                    ReportTemplateField(
                        field_id="full_name",
                        field_name="Full Name",
                        field_type=FieldType.TEXT,
                        description="Complete legal name of the person",
                        required=True,
                        order_index=1
                    ),
                    ReportTemplateField(
                        field_id="alternate_names",
                        field_name="Alternate Names",
                        field_type=FieldType.LIST,
                        description="Other spellings, nicknames, aliases, maiden names",
                        order_index=2
                    ),
                    ReportTemplateField(
                        field_id="date_of_birth",
                        field_name="Date of Birth",
                        field_type=FieldType.DATE,
                        description="Birth date",
                        order_index=3
                    ),
                    ReportTemplateField(
                        field_id="place_of_birth",
                        field_name="Place of Birth",
                        field_type=FieldType.ADDRESS,
                        description="Birth location",
                        order_index=4
                    ),
                    ReportTemplateField(
                        field_id="citizenship",
                        field_name="Citizenship(s)",
                        field_type=FieldType.LIST,
                        description="Current citizenships held",
                        order_index=5
                    ),
                    ReportTemplateField(
                        field_id="national_ids",
                        field_name="Personal National IDs",
                        field_type=FieldType.STRUCTURED_DATA,
                        description="Type, Country, ID Numbers",
                        order_index=6
                    ),
                    ReportTemplateField(
                        field_id="languages",
                        field_name="Languages Used/Known",
                        field_type=FieldType.LIST,
                        description="Languages the person speaks or knows",
                        order_index=7
                    )
                ]
            ),
            ReportTemplateSection(
                section_id="contact_information",
                section_name="Contact Information",
                description="Phone numbers, addresses, email addresses",
                order_index=2,
                fields=[
                    ReportTemplateField(
                        field_id="home_addresses",
                        field_name="Home Addresses",
                        field_type=FieldType.STRUCTURED_DATA,
                        description="Current and past home addresses with dates",
                        order_index=1
                    ),
                    ReportTemplateField(
                        field_id="phone_numbers",
                        field_name="Telephone Numbers",
                        field_type=FieldType.STRUCTURED_DATA,
                        description="Home, work, mobile numbers",
                        order_index=2
                    ),
                    ReportTemplateField(
                        field_id="email_addresses",
                        field_name="Email Addresses",
                        field_type=FieldType.STRUCTURED_DATA,
                        description="Personal and work email addresses",
                        order_index=3
                    )
                ]
            ),
            ReportTemplateSection(
                section_id="professional_background",
                section_name="Professional Background",
                description="Career history, employment, business affiliations",
                order_index=3,
                fields=[
                    ReportTemplateField(
                        field_id="current_profession",
                        field_name="Current Profession",
                        field_type=FieldType.TEXT,
                        description="Current job title and role",
                        order_index=1
                    ),
                    ReportTemplateField(
                        field_id="past_professions",
                        field_name="Past Professions",
                        field_type=FieldType.LIST,
                        description="Previous job titles and roles",
                        order_index=2
                    ),
                    ReportTemplateField(
                        field_id="company_affiliations",
                        field_name="Company Names / Addresses",
                        field_type=FieldType.STRUCTURED_DATA,
                        description="Current and past companies, dates, addresses",
                        order_index=3
                    )
                ]
            ),
            ReportTemplateSection(
                section_id="education",
                section_name="Education",
                description="Educational background and qualifications",
                order_index=4,
                fields=[
                    ReportTemplateField(
                        field_id="education_history",
                        field_name="Educational Background",
                        field_type=FieldType.STRUCTURED_DATA,
                        description="Schools, dates, majors, locations",
                        order_index=1
                    )
                ]
            ),
            ReportTemplateSection(
                section_id="social_media",
                section_name="Social Media and Online Accounts",
                description="Online presence and social media profiles",
                order_index=5,
                fields=[
                    ReportTemplateField(
                        field_id="social_accounts",
                        field_name="Social Media Accounts",
                        field_type=FieldType.STRUCTURED_DATA,
                        description="Platform names and userids/usernames",
                        order_index=1
                    )
                ]
            ),
            ReportTemplateSection(
                section_id="family_associates",
                section_name="Family Members and Associates",
                description="Family relationships and known associates",
                order_index=6,
                fields=[
                    ReportTemplateField(
                        field_id="family_members",
                        field_name="Family Members",
                        field_type=FieldType.STRUCTURED_DATA,
                        description="Spouse, children, parents, siblings with relationships",
                        order_index=1
                    ),
                    ReportTemplateField(
                        field_id="associates",
                        field_name="Associates",
                        field_type=FieldType.STRUCTURED_DATA,
                        description="Known associates and how they are known",
                        order_index=2
                    )
                ]
            ),
            ReportTemplateSection(
                section_id="affiliations",
                section_name="Groups/Affiliations",
                description="Organizational memberships and affiliations",
                order_index=7,
                fields=[
                    ReportTemplateField(
                        field_id="group_memberships",
                        field_name="Group Memberships",
                        field_type=FieldType.STRUCTURED_DATA,
                        description="Organizations, membership levels, roles",
                        order_index=1
                    )
                ]
            ),
            ReportTemplateSection(
                section_id="additional_information",
                section_name="Other Information of Interest",
                description="Hobbies, interests, newsworthy information",
                order_index=8,
                fields=[
                    ReportTemplateField(
                        field_id="interests_hobbies",
                        field_name="Interests and Hobbies",
                        field_type=FieldType.LIST,
                        description="Personal interests, hobbies, notable activities",
                        order_index=1
                    ),
                    ReportTemplateField(
                        field_id="newsworthy_info",
                        field_name="Newsworthy Information",
                        field_type=FieldType.LIST,
                        description="Notable news mentions, achievements, controversies",
                        order_index=2
                    )
                ]
            ),
            ReportTemplateSection(
                section_id="watchlist_derogatory",
                section_name="Watchlist and Derogatory Information",
                description="Security-related information and negative findings",
                order_index=9,
                fields=[
                    ReportTemplateField(
                        field_id="watchlist_info",
                        field_name="Watchlist Information",
                        field_type=FieldType.LONG_TEXT,
                        description="Any watchlist appearances or security-related information",
                        order_index=1
                    ),
                    ReportTemplateField(
                        field_id="derogatory_info",
                        field_name="Derogatory Information",
                        field_type=FieldType.LONG_TEXT,
                        description="Negative information, legal issues, controversies",
                        order_index=2
                    )
                ]
            )
        ],
        created_by="system",
        metadata={
            "based_on": "Professional OSINT and business intelligence methodologies",
            "use_case": "Business due diligence, professional assessment, and public information analysis",
            "research_methodology": "Open Source Intelligence (OSINT) using publicly available sources",
            "compliance_note": "All information gathered from publicly accessible, legally obtainable sources"
        }
    )
