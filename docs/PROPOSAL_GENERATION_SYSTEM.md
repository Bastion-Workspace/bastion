# Proposal Generation System Documentation

## Overview

The proposal generation system follows the **fiction pattern** for reference cascading:
- **proposal** document references **pr-outline** in frontmatter
- **pr-outline** contains all references (pr-req, pr-style, company_knowledge) in its frontmatter
- **pr-outline** defines sections with high-level guidance
- References cascade from pr-outline to proposal (like outline → manuscript in fiction)

## Document Types

### type: pr-outline

**Purpose:** Proposal structure and planning document (like fiction outline)

**Contains:**
- References to pr-req, pr-style, company_knowledge in frontmatter
- Section definitions with high-level guidance in body
- Structure/template information

**Example Frontmatter:**
```yaml
---
type: pr-outline
proposal_type: technical_commercial
pr_req_document_id: doc_12345  # RFI/RFP document
pr_style_document_id: doc_23456  # Style template
company_knowledge_id: doc_45678  # Company facts
document_version: 1.0
---

## Executive Summary

A compelling 1-2 paragraph summary that captures the essence of the proposed solution.
Should highlight key value propositions and customer benefits.

## Company Overview

Background, experience, and key qualifications of the company.
Include relevant case studies and certifications.

## Understanding of Requirement

Detailed understanding of the customer's needs and requirements.
Demonstrate that you've carefully analyzed their RFI/RFP.

## Proposed Solution

Specific solution approach and methodology.
Address technical and commercial aspects.

## Implementation Approach

Step-by-step implementation plan with clear phases and milestones.

## Timeline

Detailed project schedule with key deliverables and dates.

## Team Qualifications

Team composition and individual qualifications relevant to this project.

## Pricing

Pricing model and financial terms. Include cost breakdown if appropriate.

## Terms and Conditions

Standard terms and any special conditions or requirements.
```

### type: pr-req

**Purpose:** RFI/RFP (Request for Proposal) requirements document

**Usage:** Upload customer's RFI/RFP document

**Example Frontmatter:**
```yaml
---
type: pr-req
customer_name: ACME Corporation
rfp_date: 2024-01-15
rfp_deadline: 2024-02-15
---

# ACME Corporation RFP 2024-Q1

## Section 1: Technical Requirements

1. System must support 10,000 concurrent users
2. Response time must be under 200ms
...
```

### type: pr-style

**Purpose:** Proposal style/template reference document

**Usage:** Upload a previous proposal or template demonstrating desired style

**Example Frontmatter:**
```yaml
---
type: pr-style
created_date: 2023-06-15
style_guide: professional_formal
tone: professional
---

# [Customer Name] Proposal

[Reference proposal content showing style and format]
```

### type: proposal

**Purpose:** Generated proposal document (output)

**Usage:** System generates this document; references pr-outline

**Example Frontmatter:**
```yaml
---
type: proposal
customer_name: ACME Corporation
proposal_id: acme-2024-q1
pr_outline: doc_outline_123  # References pr-outline (cascades to all references)
status: draft
completeness_score: 92.5
word_count: 8250
sections:
  - executive_summary
  - company_overview
  - proposed_solution
  - pricing
last_updated: 2024-01-20T15:30:00Z
---

# ACME Corporation Proposal

[Generated content]
```

## Reference Cascading Pattern

**Like Fiction System:**
- **Fiction:** manuscript → outline → (rules, style, characters)
- **Proposal:** proposal → pr-outline → (pr-req, pr-style, company_knowledge)

**Workflow:**
1. Proposal frontmatter has `pr_outline: doc_id`
2. Agent loads pr-outline document
3. Agent extracts section definitions from pr-outline body
4. Agent cascades to pr-outline's frontmatter references:
   - `pr_req_document_id` → loads RFI/RFP
   - `pr_style_document_id` → loads style guide
   - `company_knowledge_id` → loads company facts
5. Agent uses section definitions from pr-outline for generation

## Section Definitions in pr-outline

The pr-outline document body should contain section definitions like:

```markdown
## Executive Summary

A compelling 1-2 paragraph summary that captures the essence of the proposed solution.
Should highlight key value propositions and customer benefits.
Focus on outcomes, not features.

## Company Overview

Background, experience, and key qualifications of the company.
Include relevant case studies and certifications.
Keep it concise but impactful.

## Proposed Solution

Specific solution approach and methodology.
Address both technical and commercial aspects.
Show how you'll solve their specific problems.
```

The agent extracts these as `section_definitions` dict:
- Key: section name (snake_case from header)
- Value: guidance text from section body

## Workflow

### 1. Create pr-outline Document

Create a document with `type: pr-outline`:

```yaml
---
type: pr-outline
proposal_type: technical_commercial
pr_req_document_id: [RFI/RFP doc ID]
pr_style_document_id: [Style guide doc ID]
company_knowledge_id: [Company facts doc ID]
---

## Executive Summary

[Your guidance for this section]

## Company Overview

[Your guidance for this section]

## Proposed Solution

[Your guidance for this section]

[Define all sections you want in the proposal]
```

### 2. Create proposal Document

Create a document with `type: proposal`:

```yaml
---
type: proposal
customer_name: ACME Corporation
pr_outline: [pr-outline doc ID]  # References pr-outline
---

```

### 3. Generate Proposal

Ask the AI: "Generate proposal response to RFI/RFP"

The system will:
- Load pr-outline
- Extract section definitions
- Cascade to pr-outline's references (pr-req, pr-style, company_knowledge)
- Parse RFI/RFP requirements
- Generate sections using pr-outline guidance
- Validate compliance

### 4. Generate Specific Sections

You can request specific sections:
- "Craft the introduction" → generates executive_summary
- "Write the pricing section" → generates pricing
- "Create the timeline" → generates timeline

## Benefits of pr-outline Pattern

1. **Single Source of Truth**: All references in one place (pr-outline)
2. **Section Guidance**: Define what should go in each section
3. **Reusability**: Same pr-outline can be used for multiple proposals
4. **Consistency**: Ensures all proposals follow same structure
5. **Cascading**: References automatically cascade (like fiction)

## Frontmatter Schema

### pr-outline Document

```yaml
type: pr-outline
proposal_type: string  # technical_commercial, commercial_services, etc.
pr_req_document_id: string  # Reference to RFI/RFP
pr_style_document_id: string  # Reference to style guide
company_knowledge_id: string  # Reference to company facts
document_version: number
```

**Body:** Section definitions with ## headers

### proposal Document

```yaml
type: proposal
customer_name: string
proposal_id: string
pr_outline: string  # Reference to pr-outline (cascades to all references)
status: string  # draft, final, submitted
completeness_score: number  # 0-100%
word_count: number
sections: [list]  # Sections included
last_updated: datetime
```

## Intent Gating

Proposal agent is auto-activated for:
- Document types: proposal, pr-outline, pr-req, pr-style
- Queries: "generate proposal", "respond to rfp", "create proposal"

## Integration

Proposal subgraphs are designed for composition into writing_assistant_agent in the future.

See llm-orchestrator/orchestrator/subgraphs/proposal/ for implementation details.
