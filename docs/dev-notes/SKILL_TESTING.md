# Skill Testing Guide

Use this document to verify that skill routing works correctly. For each skill, run the listed prompts and confirm the orchestrator selects the expected skill (check logs for "Skill selection:" or "Skill discovery:").

**How to test**

- **Web UI:** Send the prompt in the chat. Ensure the correct document type is open (or no document) as noted.
- **Editor-gated skills:** Open a document of the type listed in "Context" before sending the prompt.
- **image_description:** Attach an image with the message.

---

## Automation Skills (no editor required unless noted)

| Skill | Context | Test prompts |
|-------|---------|--------------|
| **weather** | None | "What's the weather in Boston?" / "Forecast for Paris next week." |
| **dictionary** | None | "Define perspicacious." / "What's the etymology of 'disaster'?" |
| **help** | None | "How do I use this app?" / "What features are available?" |
| **email** | None | "Check my inbox." / "Search emails for 'invoice'." / "Send an email to Jane: meeting tomorrow at 3." |
| **navigation** | None | "List my saved locations." / "Route from home to the office." / "Save a new location: Central Park, NYC." |
| **rss** | None | "List my RSS feeds." / "Add this feed: https://example.com/feed.xml." / "Refresh my news feed." |
| **entertainment** | None | "Recommend a movie like Inception." / "What should I read next if I liked Dune?" |
| **org_capture** | None | "Capture to my inbox: Article on the difference between book banning and protecting children." / "Add to inbox: buy milk." / "Quick capture: call dentist Friday." |
| **org_content** | **Org file open** | "List my TODOs." / "What's on my agenda today?" / "Find headings tagged :work:." |
| **image_generation** | None | "Draw a cat wearing a top hat." / "Create an image of a sunset over mountains." / "Generate a picture of a robot watering plants." |
| **image_description** | **Image attached** | "Describe this image." / "What's in this picture?" / "Caption this photo." |
| **reference** | **Reference doc open** (journal/log) | "Graph my weight log." / "Calculate 120 * 0.85." / "Convert 5 miles to km." |
| **document_creator** | None (or any; not editing current doc) | "Create a new file in Reference called project_notes.md with a summary of our meeting." / "Save this to a new document in my Projects folder." |

---

## Research Skills

| Skill | Context | Test prompts |
|-------|---------|--------------|
| **research** | None | "How do I grow tomatoes in containers?" / "What is the capital of Bhutan?" / "Do we have any comics with dragons?" / "Find me a photo of the beach from last year." |
| **content_analysis** | **Article/blog/substack/outline/reference/doc open** | "Summarize this document." / "Compare this post to my draft in Reference." / "Find conflicts between this and the outline." |
| **knowledge_builder** | None | "Fact-check the claim that vitamin C cures colds." / "Verify these statistics and build a one-page knowledge doc." / "Distill what we know about CRISPR into a short document." |
| **security_analysis** | None | "Security scan https://example.com." / "Check this website for vulnerabilities." |
| **site_crawl** | None | "Crawl example.com and extract the main content." / "One-off crawl of this domain." |
| **website_crawler** | None | "Ingest this URL and save the content." / "Recursively crawl this website and process it." |

---

## Conversational Skills

| Skill | Context | Test prompts |
|-------|---------|--------------|
| **chat** | None | "Hi!" / "What's your name?" / "Tell me a joke." / Any vague or non-specialized query. |
| **story_analysis** | None (no manuscript open, or general theory) | "Discuss the three-act structure." / "What's a good way to write dialogue?" / "Give me writing advice for pacing." |

**Do not use story_analysis for:** "Review chapter 7" or "How is this scene?" when a fiction manuscript is open — that should route to **fiction_editing**.

---

## Editor Skills (document type must be open)

| Skill | Editor type | Test prompts |
|-------|-------------|--------------|
| **fiction_editing** | Fiction manuscript | "Review chapter 3 for redundancy." / "Generate chapter 5." / "Fix the dialogue in this scene." / "How is the pacing in act 2?" |
| **outline_editing** | Outline | "Add a plot point for the midpoint." / "Restructure act 2." / "Build an outline for a mystery short story." |
| **character_development** | Character doc | "Add a flaw to the protagonist." / "Update this character sheet with the new backstory." |
| **rules_editing** | Rules/worldbuilding | "Add a rule for how magic works in this world." / "Edit the canon entry for the northern kingdoms." |
| **style_editing** | Style guide | "Update the voice guide with a no-adverbs rule." / "Edit the style doc for dialogue formatting." |
| **series_editing** | Series doc | "Update the series synopsis." / "Mark book 2 as in progress." |
| **electronics** | Electronics project | "Add a circuit for an LED with a resistor." / "What pins does the ESP32 use for I2C?" |
| **general_project** | Project plan | "Add a task: user testing by March." / "Update the scope with the new requirements." |
| **podcast_script** | Podcast doc | "Expand the intro for this episode." / "Add show notes for the guest segment." |
| **proofreading** | Fiction, outline, article, blog, substack | "Proofread this." / "Fix grammar and typos." / "Check spelling." |
| **article_writing** | Article, Substack, or blog | "Write a stronger lede." / "Edit this post for tone." / "Add a section on takeaways." |

---

## Quick routing checks

- **"Capture to my inbox: X"** → org_capture (not research).
- **"What's the weather in Tokyo?"** → weather.
- **"Do we have any images of X?"** (local collection) → research (not entertainment, not rss).
- **"Recommend a movie"** → entertainment (not research).
- **"Add this RSS feed"** or "List my feeds" → rss (not research).
- **"Create a new file in Folder Y"** → document_creator (not fiction_editing).
- **"Generate chapter 4"** with fiction doc open → fiction_editing (not document_creator).
- **"Discuss story structure"** with no manuscript open → story_analysis; **"Review this chapter"** with manuscript open → fiction_editing.

---

## Editor type reference

When testing editor skills, open a document whose frontmatter `type` (or inferred type) matches:

- **fiction** — manuscript
- **outline** — story outline
- **character** — character sheet
- **rules** — worldbuilding/canon
- **style** — style/voice guide
- **series** — series synopsis
- **electronics** — electronics project
- **project** — project plan
- **podcast** — podcast script
- **article** / **blog** / **substack** — article or post
- **org** — org-mode file (for org_content)
- **reference** — journal, log, or reference doc (for reference, content_analysis)
