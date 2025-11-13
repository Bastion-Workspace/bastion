# Research Agent Workflow Scenarios

## Complete Workflow Paths

### Scenario 1: Cache Hit (Fastest)
```
Cache Check → Cache Hit → Final Synthesis → END
```
**When**: Previous research found in conversation cache
**Result**: Instant response using cached research

---

### Scenario 2: Round 1 Sufficient (Fast)
```
Cache Check → Query Expansion → Round 1 Local → Assess Round 1 → [Sufficient] → Final Synthesis → END
```
**When**: Round 1 local search finds comprehensive, relevant results
**Result**: Quick answer from local documents only

---

### Scenario 3: Empty Local Results (Fast Path)
```
Cache Check → Query Expansion → Round 1 Local → Assess Round 1 → [Empty/No Relevant] 
  → Web Round 1 → Assess Web Round 1 → [Sufficient] → Final Synthesis → END
```
**When**: No local documents exist on the topic
**Optimization**: Skips Round 2 Local (saves API calls)
**Result**: Web search only

---

### Scenario 4: Empty Local + Web Round 2 Needed
```
Cache Check → Query Expansion → Round 1 Local → Assess Round 1 → [Empty/No Relevant] 
  → Web Round 1 → Assess Web Round 1 → [Insufficient] 
  → Gap Analysis Web → [Round 2 Needed] 
  → Web Round 2 → Final Synthesis → END
```
**When**: No local docs, Web Round 1 insufficient
**Result**: Two-round web search for depth

---

### Scenario 5: Some Local Results + Severe Gaps (Optimized)
```
Cache Check → Query Expansion → Round 1 Local → Assess Round 1 → [Insufficient] 
  → Gap Analysis → [Severe + Needs Web] 
  → Web Round 1 → Assess Web Round 1 → [Sufficient] → Final Synthesis → END
```
**When**: Local docs exist but have severe gaps, gap analysis says web needed
**Optimization**: Skips Round 2 Local (saves API calls)
**Result**: Local Round 1 + Web Round 1

---

### Scenario 6: Some Local Results + Minor Gaps (Deep Path)
```
Cache Check → Query Expansion → Round 1 Local → Assess Round 1 → [Insufficient] 
  → Gap Analysis → [Minor/Moderate] 
  → Round 2 Local → Assess Round 2 → [Sufficient] → Final Synthesis → END
```
**When**: Local docs exist with minor gaps
**Result**: Two-round local search, no web needed

---

### Scenario 7: Some Local Results + Minor Gaps + Web Needed
```
Cache Check → Query Expansion → Round 1 Local → Assess Round 1 → [Insufficient] 
  → Gap Analysis → [Minor/Moderate] 
  → Round 2 Local → Assess Round 2 → [Insufficient] 
  → Web Round 1 → Assess Web Round 1 → [Sufficient] → Final Synthesis → END
```
**When**: Local docs exist but Round 2 Local still insufficient
**Result**: Two-round local + Web Round 1

---

### Scenario 8: Full Deep Research (Most Comprehensive)
```
Cache Check → Query Expansion → Round 1 Local → Assess Round 1 → [Insufficient] 
  → Gap Analysis → [Minor/Moderate] 
  → Round 2 Local → Assess Round 2 → [Insufficient] 
  → Web Round 1 → Assess Web Round 1 → [Insufficient] 
  → Gap Analysis Web → [Round 2 Needed] 
  → Web Round 2 → Final Synthesis → END
```
**When**: Complex query requiring maximum depth
**Result**: Two-round local + Two-round web = Maximum coverage

---

## Edge Cases & Error Handling

### Edge Case 1: Round 1 Assessment Fails
**Behavior**: Falls back to `sufficient=False, has_relevant_info=False`
**Result**: Proceeds to gap analysis (safe default)

### Edge Case 2: Gap Analysis Parsing Fails
**Behavior**: Falls back to original query as gap, assumes web search needed
**Result**: Proceeds to Web Round 1 (safe default)

### Edge Case 3: Web Round 1 Fails (Network Error)
**Behavior**: Returns error in `web_round1_results`
**Result**: Assessment marks as insufficient, may try Round 2 Web or synthesize with available local results

### Edge Case 4: Web Round 1 Assessment Fails
**Behavior**: Falls back to `sufficient=False`
**Result**: Proceeds to web gap analysis (safe default)

### Edge Case 5: Web Gap Analysis Parsing Fails
**Behavior**: Falls back to `needs_web_round2=False`
**Result**: Proceeds to synthesis (conservative - avoids unnecessary Round 2)

### Edge Case 6: Round 2 Local Finds Nothing
**Behavior**: `round2_sufficient=False` (has_results check fails)
**Result**: Proceeds to Web Round 1

### Edge Case 7: Mixed Results (Some Local, Some Web)
**Behavior**: Synthesis combines all available sources
**Result**: Comprehensive answer using both local and web sources

### Edge Case 8: All Searches Return Empty
**Behavior**: Synthesis receives minimal context
**Result**: LLM acknowledges limited information and provides best possible answer

---

## Decision Points Summary

| Decision Point | Condition | Action |
|---------------|-----------|--------|
| **After Round 1** | Sufficient | → Synthesize |
| **After Round 1** | Empty/No Relevant | → Skip Round 2 Local, go to Web Round 1 |
| **After Round 1** | Insufficient | → Gap Analysis |
| **After Gap Analysis** | Severe + Needs Web | → Skip Round 2 Local, go to Web Round 1 |
| **After Gap Analysis** | Minor/Moderate | → Round 2 Local |
| **After Round 2 Local** | Sufficient | → Synthesize |
| **After Round 2 Local** | Insufficient | → Web Round 1 |
| **After Web Round 1** | Sufficient | → Synthesize |
| **After Web Round 1** | Insufficient | → Web Gap Analysis |
| **After Web Gap Analysis** | Round 2 Needed | → Web Round 2 |
| **After Web Gap Analysis** | Round 2 Not Needed | → Synthesize |

---

## Performance Characteristics

### Fastest Paths (1-2 API calls)
- Cache Hit: 0 searches
- Round 1 Sufficient: 1 local search + 1 assessment
- Empty → Web Round 1 Sufficient: 1 local search + 1 web search + 1 assessment

### Medium Paths (3-5 API calls)
- Round 2 Local Sufficient: 2 local searches + 2 assessments + 1 gap analysis
- Empty → Web Round 2: 1 local search + 2 web searches + 2 assessments + 1 gap analysis

### Deepest Path (6-8 API calls)
- Full Deep Research: 2 local searches + 2 web searches + 4 assessments + 2 gap analyses

---

## Optimization Benefits

### Before Optimization
- **Always did Round 2 Local** even when local docs don't exist
- **Wasted API calls** on empty Round 2 searches
- **Single web search** - no depth for complex queries

### After Optimization
- **Early exit** when Round 1 is empty/no relevant
- **Gap severity routing** skips Round 2 Local when severe gaps detected
- **Two-round web search** for comprehensive coverage
- **Smart routing** based on gap analysis intelligence
- **Assessment checkpoints** at every stage for early exits

**Estimated API Call Reduction**: 20-40% for queries with no local documents

