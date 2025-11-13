# Fiction Agent: Clarifying Questions for Quality

**BULLY!** The agent now asks thoughtful questions when needed for top-quality narration!

---

## Change Summary

**What:** Added `clarifying_questions` field to `ManuscriptEdit` model and enhanced Fiction Agent prompt to ask questions when genuinely needed for quality.

**Why:** Some requests are ambiguous or have multiple valid creative directions. Author input ensures the LLM produces exactly what's wanted.

**When:** Agent asks questions ONLY when necessary, not for trivial or obvious matters.

---

## The Problem

**Before:**
```
User: "Add more tension"

Agent: *Adds generic tension without knowing:*
- Internal tension (character anxiety)?
- External tension (threat appears)?
- Subtle building tension or immediate crisis?
- Should it escalate or simmer?
```

**Result:** Generic edit that might not match author's vision.

---

## The Solution

**Now:**
```
User: "Add more tension"

Agent Response:
"Add tension - should this be internal (Fleet's anxiety) or external (a threat appears)?"

[Can still provide tentative edit while asking]
```

**Result:** Targeted question leads to precise, high-quality edit aligned with author intent.

---

## Model Changes

### ManuscriptEdit Model Enhancement

**File:** `backend/models/agent_response_models.py`

**Added field:**
```python
class ManuscriptEdit(BaseModel):
    """Structured, validated edit plan for fiction manuscript changes."""
    target_filename: str
    operations: List[EditorOperation]
    scope: Literal["paragraph", "chapter", "multi_chapter"]
    chapter_index: Optional[int]
    safety: Literal["low", "medium", "high"]
    summary: str
    clarifying_questions: Optional[List[str]] = Field(
        default=None,
        description="Questions to ask the user for clarification when request is ambiguous or requires author input for quality"
    )
```

**Key features:**
- Optional (None if no questions needed)
- List of strings (can ask multiple questions)
- Integrated into structured output validation

---

## System Prompt Guidance

### When to ASK Questions

The agent asks questions when:

1. **Request is genuinely ambiguous**
   - "Make it better" - better how?
   - "Add more description" - of what specifically?
   - "Improve pacing" - faster or slower?

2. **Multiple valid creative directions exist**
   - Tone choice (serious vs. humorous)
   - POV shift implications
   - Pacing approach (gradual vs. sudden)
   - Character motivation options

3. **Author preference would significantly impact quality**
   - Revelation should be subtle or dramatic?
   - Confrontation escalates to violence or stays verbal?
   - Foreshadowing should be obvious or hidden?

4. **Continuity issue detected**
   - "Chapter 3 shows Peterson trusting Fleet, but this revision makes him suspicious. Should I adjust Chapter 3 for consistency?"
   - "Outline says 'Fleet discovers truth' but doesn't specify what truth. What should he discover?"

5. **Request could damage story logic**
   - Adding element that contradicts established rules
   - Character behaving inconsistently without explanation
   - Plot hole would be created

6. **Character motivation unclear**
   - Why would character do this?
   - What's driving this emotional shift?
   - What's the character's goal in this scene?

---

### When NOT to Ask Questions

The agent does NOT ask when:

1. **Request is clear enough to execute well**
   - "End Chapter 1 with Peterson's witty thought"
   - "Revise the dialogue to be more formal"
   - "Add a scene transition after the confrontation"

2. **Style guide/rules already provide guidance**
   - Tone is established in style guide
   - Character behavior defined in character docs
   - Universe rules answer the question

3. **Question would be trivial or obvious**
   - "Should I use good grammar?" (obvious)
   - "What tense should I use?" (obvious from manuscript)
   - "Should I add a comma?" (trivial)

4. **Can make reasonable inference from context**
   - Existing chapters show clear tone
   - Character has established personality
   - Pacing pattern is evident

5. **Minor stylistic choices that don't affect quality**
   - Word choice variations
   - Sentence structure preferences
   - Punctuation style

---

## Question Quality Standards

### ✅ GOOD Questions

**Specific and actionable:**
```
- "Should the confrontation escalate to physical violence, or stay verbal?"
- "Add tension - should this be internal (Fleet's anxiety) or external (a threat appears)?"
- "Should this revelation be subtle foreshadowing or direct reveal?"
```

**Offer options when helpful:**
```
- "The outline says 'Fleet discovers the truth' but doesn't specify what truth. What should he discover?"
- "Should Peterson's reaction be immediate anger or slow-building resentment?"
```

**Reference specific text/context:**
```
- "Chapter 3 shows Peterson trusting Fleet, but this revision makes him suspicious. Should I adjust Chapter 3 for consistency?"
- "The outline shows Fleet investigating the merchants, but Chapter 2 established he's focused on the yacht club. Which takes precedence?"
```

**Limited in number (max 2-3):**
```json
{
  "clarifying_questions": [
    "Should this revelation be subtle or dramatic?",
    "Chapter 3 contradicts this - adjust both chapters or just current?"
  ]
}
```

---

### ❌ BAD Questions

**Too vague or unhelpful:**
```
- "What do you want me to do?"
- "How should I write this?"
- "Is this okay?"
```

**Too obvious:**
```
- "Should I use good grammar?"
- "Do you want complete sentences?"
- "Should I match the existing style?"
```

**Too trivial:**
```
- "Should I capitalize proper nouns?"
- "Should I add a comma here?"
- "What font should I use?"
```

**Already answered by context:**
```
- "What tense is the story?" (obvious from manuscript)
- "Who is the protagonist?" (obvious from content)
- "What genre is this?" (obvious from style)
```

---

## Response Handling

### Agent Can Still Provide Tentative Edit

**Important:** Questions don't block the edit. Agent can:
- Ask questions AND provide best-guess edit
- Ask questions AND leave operations empty (if too uncertain)
- Ask questions AND note assumptions in summary

**Example:**
```json
{
  "clarifying_questions": [
    "Should this revelation be subtle or dramatic?",
    "Does Chapter 3 need adjustment for consistency?"
  ],
  "operations": [{
    "op_type": "replace_range",
    "original_text": "...",
    "text": "... [subtle version, can revise if you want dramatic]"
  }],
  "summary": "Added revelation (subtle approach). Can make more dramatic if preferred."
}
```

---

### User Experience

**User sees:**
```
[Generated prose preview OR summary]

**Questions for clarification:**
- Should the confrontation escalate to physical violence, or stay verbal?
- Chapter 3 contradicts this change - adjust both chapters or just current?
```

**User can:**
1. Answer the questions and resubmit
2. Accept the tentative edit as-is
3. Modify the request based on questions
4. Provide more context

---

## Example Scenarios

### Scenario 1: Ambiguous Request

**User:** "Make the ending better"

**Agent Response:**
```json
{
  "clarifying_questions": [
    "Better how? More dramatic, more emotional, more surprising, or something else?",
    "Should it end on a cliffhanger or provide resolution?"
  ],
  "operations": [],
  "summary": "Awaiting clarification on ending direction before proceeding."
}
```

**User sees:**
```
Awaiting clarification on ending direction before proceeding.

**Questions for clarification:**
- Better how? More dramatic, more emotional, more surprising, or something else?
- Should it end on a cliffhanger or provide resolution?
```

---

### Scenario 2: Continuity Issue Detected

**User:** "Revise Peterson's reaction to show more skepticism"

**Agent analyzes adjacent chapters:**
- Chapter 1: Peterson is trusting
- Chapter 3: References Peterson's trust as motivation

**Agent Response:**
```json
{
  "clarifying_questions": [
    "Chapter 1 shows Peterson trusting, and Chapter 3 references this trust. Should I adjust both chapters for consistency, or is this a character arc shift in Chapter 2 only?"
  ],
  "operations": [{
    "op_type": "replace_range",
    "original_text": "Peterson nodded thoughtfully.",
    "text": "Peterson's eyes narrowed. \"I'm not convinced,\" he said, skepticism clear in his voice."
  }],
  "summary": "Added skepticism to Peterson's reaction. Note: May require Chapter 1 and 3 adjustments for consistency."
}
```

**User sees:**
```
[Preview of revised dialogue with skepticism]

**Questions for clarification:**
- Chapter 1 shows Peterson trusting, and Chapter 3 references this trust. Should I adjust both chapters for consistency, or is this a character arc shift in Chapter 2 only?
```

---

### Scenario 3: Multiple Valid Approaches

**User:** "Add more tension to the scene"

**Agent Response:**
```json
{
  "clarifying_questions": [
    "Should this be internal tension (Fleet's anxiety, doubts) or external tension (threat, danger appears)?",
    "Gradual build-up or immediate spike in tension?"
  ],
  "operations": [{
    "op_type": "replace_range",
    "original_text": "Fleet watched the approaching yacht.",
    "text": "Fleet watched the approaching yacht, his hand instinctively moving to his pocket. Something felt wrong."
  }],
  "summary": "Added subtle internal tension (Fleet's instincts). Can adjust to external threat or increase intensity if preferred."
}
```

**User sees:**
```
Fleet watched the approaching yacht, his hand instinctively moving to his pocket. Something felt wrong.

**Questions for clarification:**
- Should this be internal tension (Fleet's anxiety, doubts) or external tension (threat, danger appears)?
- Gradual build-up or immediate spike in tension?
```

---

### Scenario 4: Clear Request (No Questions)

**User:** "End Chapter 1 with a witty thought from Peterson"

**Agent Response:**
```json
{
  "clarifying_questions": null,
  "operations": [{
    "op_type": "insert_after_heading",
    "anchor_text": "But a billion dollars is an extraordinary amount to spend on nostalgia alone.",
    "text": "\n\nI nodded, already making mental notes about what to pack. If Fleet was wrong, at worst I'd enjoy a week on a luxury yacht. If he was right... well, someone had to keep him from getting himself killed. Again."
  }],
  "summary": "Added witty ending to Chapter 1 from Peterson's perspective."
}
```

**User sees:**
```
I nodded, already making mental notes about what to pack. If Fleet was wrong, at worst I'd enjoy a week on a luxury yacht. If he was right... well, someone had to keep him from getting himself killed. Again.
```

**No questions!** Request was clear, context was sufficient, agent executed confidently.

---

## Implementation Details

### Prompt Section

**Added to system prompt:**
```
=== CLARIFYING QUESTIONS (clarifying_questions field) ===

**ASK questions when:**
- Request is genuinely ambiguous ("make it better" - better how?)
- Multiple valid creative directions exist (tone choice, POV, pacing approach)
- Author preference would significantly impact quality
- Continuity issue detected that needs author decision
- Request could damage story logic without clarification
- Character motivation unclear from context

**DO NOT ask questions when:**
- Request is clear enough to execute well
- Style guide/rules already provide guidance
- Question would be trivial or obvious
- You can make a reasonable inference from context
- Minor stylistic choices that don't affect quality

**Question quality standards:**
- Specific and actionable (not "What do you want?")
- Offer options when helpful
- Reference specific text/context
- Maximum 2-3 questions (don't overwhelm)
- If asking questions, you can still provide a tentative operation

**Examples of GOOD questions:**
- "Should the confrontation escalate to physical violence, or stay verbal?"
- "Chapter 3 shows Peterson trusting Fleet, but this revision makes him suspicious. Should I adjust Chapter 3?"
- "The outline says 'Fleet discovers the truth' but doesn't specify what truth. What should he discover?"

**Examples of BAD questions:**
- "Should I use good grammar?" (obvious)
- "What do you want me to do?" (too vague)
- "What tense should I use?" (obvious from manuscript)
```

---

### Response Processing

**File:** `backend/services/langgraph_agents/fiction_editing_agent.py`

**Code:**
```python
response_text = generated_preview if generated_preview else (structured.summary or "Edit plan ready.")

# Add clarifying questions to response if present
clarifying_questions = getattr(structured, "clarifying_questions", None)
if clarifying_questions and len(clarifying_questions) > 0:
    questions_section = "\n\n**Questions for clarification:**\n" + "\n".join([
        f"- {q}" for q in clarifying_questions
    ])
    response_text = response_text + questions_section
    logger.info(f"❓ FICTION AGENT: Asking {len(clarifying_questions)} clarifying question(s)")
```

**Result:**
- Questions appended to response text
- Formatted as markdown list
- Logged for visibility
- User sees questions in chat UI

---

## Benefits

### 1. **Better Quality Output**
- Agent gets needed information for precise edits
- No more generic "make it better" attempts
- Author vision accurately captured

### 2. **Fewer Regenerations**
- Get it right the first time with clarification
- No back-and-forth trying to guess intent
- Saves tokens and time

### 3. **Continuity Awareness**
- Agent can flag cross-chapter issues
- User decides scope of changes
- Prevents accidental plot holes

### 4. **Creative Collaboration**
- Agent respects multiple valid approaches
- Author makes creative decisions
- LLM executes technical implementation

### 5. **Thoughtful, Not Annoying**
- Only asks when genuinely needed
- Offers options, not vague "what do you want?"
- Maximum 2-3 questions to avoid overwhelm

---

## Roosevelt's Philosophy

**Trust the LLM's judgment on WHEN to ask:**
- Agent decides if question is needed
- No hardcoded triggers for questioning
- LLM uses semantic understanding of ambiguity
- Balances asking vs. proceeding confidently

**Like a cavalry scout:**
- Most of the time, charge forward confidently
- When genuinely uncertain, ask for orders
- When multiple routes exist, consult the general
- When trap detected, warn before proceeding

**By George!** A smart agent knows when to ask and when to act!

---

## Testing Scenarios

### Test 1: Ambiguous Request
```
User: "Improve the prose"
Expected: Questions about what aspect to improve (pacing, description, dialogue, etc.)
```

### Test 2: Clear Request
```
User: "End Chapter 1 with Peterson's witty thought"
Expected: No questions, direct execution
```

### Test 3: Continuity Issue
```
User: "Make Peterson more suspicious"
(Chapter 3 shows Peterson trusting Fleet)
Expected: Question about whether to adjust Chapter 3 for consistency
```

### Test 4: Multiple Approaches
```
User: "Add tension"
Expected: Question about internal vs external tension, gradual vs sudden
```

### Test 5: Outline Ambiguity
```
Outline: "Fleet discovers the truth"
User: "Write the discovery scene"
Expected: Question about what truth Fleet discovers
```

---

## Files Modified

**1. `backend/models/agent_response_models.py`**
- Added `clarifying_questions: Optional[List[str]]` field to `ManuscriptEdit` model

**2. `backend/services/langgraph_agents/fiction_editing_agent.py`**
- Added "CLARIFYING QUESTIONS" section to system prompt (lines 264-304)
- Added question handling in response processing (lines 821-828)

---

## Roosevelt's Verdict

**BULLY!** The cavalry now knows when to ask for orders!

**Key principles:**
- ✅ Ask when genuinely needed for quality
- ✅ Don't ask trivial or obvious questions
- ✅ Offer options and reference context
- ✅ Limit to 2-3 questions maximum
- ✅ Can still provide tentative edit while asking
- ✅ LLM decides when to ask (no hardcoded rules)

**By George!** A thoughtful question prevents a wasted charge! But don't ask permission to breathe! 🏇

---

**Last Updated:** October 29, 2025  
**Status:** Active - Clarifying questions enabled for quality  
**Philosophy:** Trust the LLM to know when to ask, teach it how to ask well







