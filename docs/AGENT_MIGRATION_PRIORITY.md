# Agent Migration Priority Analysis

## Current Migration Status

### ✅ Already Migrated (12 agents)
1. **ResearchAgent** (FullResearchAgent) - Multi-round sophisticated research
2. **ChatAgent** (524 lines) - Conversational interactions
3. **DataFormattingAgent** - Data structuring and formatting
4. **HelpAgent** - Application help and documentation
5. **WeatherAgent** (450 lines + tools) - Weather forecasts and conditions
6. **ImageGenerationAgent** (128 lines) - DALL-E/Gemini image generation
7. **FactCheckingAgent** (473 lines) - Automated fact verification via web search
8. **RSSAgent** (632 lines) - RSS feed management via natural language
9. **OrgInboxAgent** (554 lines + 9 gRPC RPCs) - Org-mode inbox management with LLM interpretation
10. **SubstackAgent** (448 lines) - Long-form article and tweet generation
11. **PodcastScriptAgent** (394 lines) - ElevenLabs TTS podcast script generation
12. **OrgProjectAgent** (492 lines) - Project capture with HITL preview-confirm workflow ✨ NEW

### 📋 Remaining Agents (12+ agents)

---

## Priority Tiers

### **TIER 1: Simple, High-Value, Frequently Used** 🎯

These agents are simpler (<500 lines), likely frequently used, and have minimal backend dependencies.

#### 1. **WeatherAgent** (~300 lines)
- **Complexity**: ⭐ LOW
- **Dependencies**: External API (OpenWeatherMap)
- **Use Case**: Weather queries
- **Backend Dependencies**: Minimal (API key only)
- **Migration Effort**: 2-3 hours
- **Value**: High (common use case)
- **Priority**: ⭐⭐⭐⭐⭐

**Why First?**
- Simple external API integration
- Clear, focused responsibility
- Good test case for external service integration
- Frequently used feature

#### 2. **ImageGenerationAgent** (~350 lines)
- **Complexity**: ⭐ LOW
- **Dependencies**: DALL-E API
- **Use Case**: Generate images
- **Backend Dependencies**: Minimal (API key only)
- **Migration Effort**: 2-3 hours
- **Value**: Medium-High (creative use case)
- **Priority**: ⭐⭐⭐⭐

**Why Second?**
- Simple external API integration
- Similar pattern to WeatherAgent
- Validates external service migration pattern

#### 3. **FactCheckingAgent** (~400 lines)
- **Complexity**: ⭐⭐ LOW-MEDIUM
- **Dependencies**: Web search tools
- **Use Case**: Verify claims and facts
- **Backend Dependencies**: Search tools (already in llm-orchestrator)
- **Migration Effort**: 3-4 hours
- **Value**: Medium
- **Priority**: ⭐⭐⭐

---

### **TIER 2: Medium Complexity, High Usage** 🎖️

These agents are moderate complexity (500-700 lines), commonly used, and provide specialized value.

#### 4. **EntertainmentAgent** (574 lines)
- **Complexity**: ⭐⭐ MEDIUM
- **Dependencies**: Web search for movie/TV info
- **Use Case**: Movie/TV recommendations and information
- **Backend Dependencies**: Search tools
- **Migration Effort**: 4-5 hours
- **Value**: Medium-High (popular feature)
- **Priority**: ⭐⭐⭐⭐

#### 5. **OrgInboxAgent** (662 lines)
- **Complexity**: ⭐⭐⭐ MEDIUM
- **Dependencies**: Org-mode file manipulation
- **Use Case**: Manage TODO items in inbox.org
- **Backend Dependencies**: File system tools, org-mode tools
- **Migration Effort**: 5-6 hours
- **Value**: High (productivity feature)
- **Priority**: ⭐⭐⭐⭐

#### 6. **OrgProjectAgent** (492 lines)
- **Complexity**: ⭐⭐ MEDIUM
- **Dependencies**: Org-mode file creation
- **Use Case**: Create structured projects
- **Backend Dependencies**: File system tools, org-mode tools
- **Migration Effort**: 4-5 hours
- **Value**: Medium (productivity feature)
- **Priority**: ⭐⭐⭐

#### 7. **WebsiteCrawlerAgent** (521 lines)
- **Complexity**: ⭐⭐ MEDIUM
- **Dependencies**: Web crawling tools
- **Use Case**: Ingest entire websites
- **Backend Dependencies**: Crawl tools (need to verify availability)
- **Migration Effort**: 5-6 hours
- **Value**: Medium (data ingestion)
- **Priority**: ⭐⭐⭐

---

### **TIER 3: Creative/Editing Agents** 📝

Fiction writing and creative development agents. Complex prompts but similar patterns.

#### 8. **ProofreadingAgent** (~400 lines)
- **Complexity**: ⭐⭐ MEDIUM
- **Dependencies**: Style guide access
- **Use Case**: Grammar and style corrections
- **Backend Dependencies**: Document access, style guide
- **Migration Effort**: 4-5 hours
- **Value**: High (quality assurance)
- **Priority**: ⭐⭐⭐⭐

#### 9. **OutlineEditingAgent** (649 lines)
- **Complexity**: ⭐⭐⭐ MEDIUM
- **Dependencies**: Editor context
- **Use Case**: Create/refine story outlines
- **Backend Dependencies**: Document access
- **Migration Effort**: 5-6 hours
- **Value**: Medium (creative writing)
- **Priority**: ⭐⭐⭐

#### 10. **RulesEditingAgent** (596 lines)
- **Complexity**: ⭐⭐⭐ MEDIUM
- **Dependencies**: Editor context
- **Use Case**: World-building rules
- **Backend Dependencies**: Document access
- **Migration Effort**: 5-6 hours
- **Value**: Medium (creative writing)
- **Priority**: ⭐⭐⭐

#### 11. **CharacterDevelopmentAgent** (695 lines)
- **Complexity**: ⭐⭐⭐ MEDIUM-HIGH
- **Dependencies**: Editor context
- **Use Case**: Develop character profiles
- **Backend Dependencies**: Document access
- **Migration Effort**: 6-7 hours
- **Value**: Medium (creative writing)
- **Priority**: ⭐⭐⭐

#### 12. **FictionEditingAgent** (944 lines)
- **Complexity**: ⭐⭐⭐⭐ HIGH
- **Dependencies**: Editor context, complex prompts
- **Use Case**: Create/edit fiction prose
- **Backend Dependencies**: Document access, style guides
- **Migration Effort**: 8-10 hours
- **Value**: High (creative writing)
- **Priority**: ⭐⭐⭐

#### 13. **StoryAnalysisAgent** (~450 lines)
- **Complexity**: ⭐⭐⭐ MEDIUM
- **Dependencies**: Editor context
- **Use Case**: Critique fiction manuscripts
- **Backend Dependencies**: Document access
- **Migration Effort**: 5-6 hours
- **Value**: Medium (creative writing)
- **Priority**: ⭐⭐⭐

---

### **TIER 4: Specialized Complex Agents** 🎓

These agents have complex logic, significant prompts, or specialized dependencies.

#### 14. **SysMLAgent** (614 lines)
- **Complexity**: ⭐⭐⭐⭐ HIGH
- **Dependencies**: Diagram generation
- **Use Case**: System design, UML, SysML diagrams
- **Backend Dependencies**: Diagram tools
- **Migration Effort**: 7-8 hours
- **Value**: Low-Medium (specialized use case)
- **Priority**: ⭐⭐

#### 17. **RSSAgent** (718 lines)
- **Complexity**: ⭐⭐⭐ MEDIUM-HIGH
- **Dependencies**: RSS feed management
- **Use Case**: Manage RSS feeds
- **Backend Dependencies**: Feed parsing, storage
- **Migration Effort**: 7-8 hours
- **Value**: Medium (information aggregation)
- **Priority**: ⭐⭐⭐

#### 18. **WargamingAgent** (923 lines)
- **Complexity**: ⭐⭐⭐⭐ HIGH
- **Dependencies**: Complex scenario analysis
- **Use Case**: Military scenario outcomes
- **Backend Dependencies**: Minimal
- **Migration Effort**: 8-10 hours
- **Value**: Low (specialized use case)
- **Priority**: ⭐⭐

#### 19. **ContentAnalysisAgent** (1470 lines)
- **Complexity**: ⭐⭐⭐⭐⭐ VERY HIGH
- **Dependencies**: Multi-document analysis
- **Use Case**: Document comparison, analysis
- **Backend Dependencies**: Document access, comparison tools
- **Migration Effort**: 12-15 hours
- **Value**: High (document management)
- **Priority**: ⭐⭐⭐⭐

---

### **TIER 5: Very Complex or Backend-Dependent** ⚙️

These agents are either very large, have significant backend dependencies, or require infrastructure changes.

#### 20. **PipelineDesignerAgent** (1887 lines)
- **Complexity**: ⭐⭐⭐⭐⭐ VERY HIGH
- **Dependencies**: Complex AWS pipeline logic
- **Use Case**: Design AWS data pipelines
- **Backend Dependencies**: Significant (AWS validation, template generation)
- **Migration Effort**: 15-20 hours
- **Value**: Medium (specialized use case)
- **Priority**: ⭐⭐

**Note**: This agent is highly specialized and may need significant refactoring for llm-orchestrator.

---

## Recommended Migration Sequence

### **Phase 1: Simple External Services** (Week 1)
1. **WeatherAgent** ⭐⭐⭐⭐⭐ (2-3 hours)
2. **ImageGenerationAgent** ⭐⭐⭐⭐ (2-3 hours)
3. **FactCheckingAgent** ⭐⭐⭐ (3-4 hours)

**Total Effort**: ~8-10 hours
**Value**: Validates external service integration pattern

### **Phase 2: Productivity & Utility** (Week 2)
4. **EntertainmentAgent** ⭐⭐⭐⭐ (4-5 hours)
5. **ProofreadingAgent** ⭐⭐⭐⭐ (4-5 hours)
~~6. **OrgInboxAgent** ⭐⭐⭐⭐ (5-6 hours)~~ ✅ MIGRATED
~~7. **OrgProjectAgent** ⭐⭐⭐ (4-5 hours)~~ ✅ MIGRATED

**Total Effort**: ~4-5 hours remaining
**Value**: High-impact productivity features

### **Phase 3: Creative Writing Suite** (Weeks 3-4)
8. **OutlineEditingAgent** ⭐⭐⭐ (5-6 hours)
9. **RulesEditingAgent** ⭐⭐⭐ (5-6 hours)
10. **CharacterDevelopmentAgent** ⭐⭐⭐ (6-7 hours)
11. **StoryAnalysisAgent** ⭐⭐⭐ (5-6 hours)
12. **FictionEditingAgent** ⭐⭐⭐ (8-10 hours)

**Total Effort**: ~29-35 hours
**Value**: Complete creative writing toolset

### **Phase 4: Content Creation** (Week 5) ✅ COMPLETE
~~13. **SubstackAgent** ⭐⭐⭐ (6-7 hours)~~ ✅ MIGRATED
~~14. **PodcastScriptAgent** ⭐⭐⭐ (6-7 hours)~~ ✅ MIGRATED
15. **WebsiteCrawlerAgent** ⭐⭐⭐ (5-6 hours)

**Total Effort**: ~5-6 hours remaining
**Value**: Content creation pipeline

### **Phase 5: Specialized & Complex** (Weeks 6-8)
16. **RSSAgent** ⭐⭐⭐ (7-8 hours)
17. **SysMLAgent** ⭐⭐ (7-8 hours)
18. **WargamingAgent** ⭐⭐ (8-10 hours)
19. **ContentAnalysisAgent** ⭐⭐⭐⭐ (12-15 hours)
20. **PipelineDesignerAgent** ⭐⭐ (15-20 hours)

**Total Effort**: ~49-61 hours
**Value**: Completes specialized capabilities

---

## Migration Patterns by Complexity

### **Pattern A: Simple External API** (Weather, Image Generation)
```
1. Port agent class (2 hours)
2. Verify API key access (30 min)
3. Test external service calls (1 hour)
4. Update gRPC routing (30 min)
Total: 2-3 hours
```

### **Pattern B: Search-Based** (Entertainment, FactChecking)
```
1. Port agent class (3 hours)
2. Verify search tool access (1 hour)
3. Test search integration (1 hour)
4. Update gRPC routing (30 min)
Total: 4-5 hours
```

### **Pattern C: File/Org-Mode** (OrgInbox, OrgProject)
```
1. Port agent class (3-4 hours)
2. Verify file system tools (1 hour)
3. Verify org-mode tools (1 hour)
4. Test file operations (1 hour)
5. Update gRPC routing (30 min)
Total: 5-7 hours
```

### **Pattern D: Editor-Context** (Fiction, Outline, Rules, Character)
```
1. Port agent class (4-6 hours)
2. Verify editor context extraction (1 hour)
3. Verify document access (1 hour)
4. Test with active editor (2 hours)
5. Update gRPC routing (30 min)
Total: 6-10 hours
```

### **Pattern E: Very Complex** (Content Analysis, Pipeline Designer)
```
1. Analyze dependencies (2-3 hours)
2. Port agent class (6-10 hours)
3. Port/adapt backend tools (3-5 hours)
4. Integration testing (2-3 hours)
5. Update gRPC routing (30 min)
Total: 12-20 hours
```

---

## Dependencies to Consider

### **External APIs**:
- OpenWeatherMap (WeatherAgent) ✅ Simple
- DALL-E/OpenAI (ImageGenerationAgent) ✅ Simple
- Web search (multiple agents) ✅ Available in llm-orchestrator

### **Backend Tools**:
- File system operations ⚠️ Need gRPC tool service
- Org-mode file parsing ⚠️ Need gRPC tool service
- Document access ⚠️ Need gRPC tool service
- Web crawling ⚠️ Need gRPC tool service

### **Context Requirements**:
- Editor context ✅ Available in proto
- Pipeline context ✅ Available in proto
- Permission grants ✅ Available in proto
- Conversation history ✅ Available in proto

---

## Success Criteria Per Agent

Each migrated agent must:
- ✅ Route correctly via intent classification
- ✅ Access necessary tools via gRPC
- ✅ Produce identical outputs to backend version
- ✅ Handle errors gracefully
- ✅ Log with same patterns as backend
- ✅ Support context (editor, pipeline, permissions)
- ✅ No lint errors
- ✅ Documentation updated

---

## Recommendation

**Start with Phase 1 (Simple External Services):**

1. **WeatherAgent** - Validates external API pattern
2. **ImageGenerationAgent** - Confirms external API pattern
3. **FactCheckingAgent** - Tests search tool integration

**These three agents:**
- Are simple and quick (2-4 hours each)
- Validate migration patterns
- Provide immediate user value
- Build confidence before tackling complex agents

**After Phase 1 success, proceed to Phase 2 (Productivity & Utility) for high-impact features.**

---

## Total Migration Effort Estimate

- **Phase 1**: ~8-10 hours
- **Phase 2**: ~17-21 hours
- **Phase 3**: ~29-35 hours
- **Phase 4**: ~17-20 hours
- **Phase 5**: ~49-61 hours

**Total**: ~120-147 hours (15-18 working days)

**With incremental rollout**: Can deploy after each phase for user feedback.

---

## Risk Assessment

### **Low Risk** ⭐⭐⭐⭐⭐
- WeatherAgent
- ImageGenerationAgent
- FactCheckingAgent

### **Medium Risk** ⭐⭐⭐
- Most editor-context agents
- OrgInbox/OrgProject
- Entertainment
- Website Crawler

### **High Risk** ⭐
- ContentAnalysisAgent (very complex)
- PipelineDesignerAgent (highly specialized)
- Agents requiring significant backend tool refactoring

---

## Next Immediate Steps

1. **Review this priority list** - Confirm Phase 1 selections
2. **Migrate WeatherAgent** - Establish external API pattern
3. **Test thoroughly** - Ensure routing and functionality work
4. **Document pattern** - Create migration template for similar agents
5. **Proceed to next agent** - Build momentum with quick wins

**The cavalry is ready to charge!** 🎖️

