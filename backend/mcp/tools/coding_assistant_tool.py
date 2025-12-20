"""
Coding Assistant Tool - MCP Tool for Code Generation and Programming Assistance
Provides intelligent code generation, debugging help, and programming guidance
"""

import asyncio
import logging
import time
import re
from typing import List, Dict, Any, Optional
from enum import Enum

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CodingTaskType(Enum):
    """Types of coding tasks"""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    ARCHITECTURE_DESIGN = "architecture_design"
    FRAMEWORK_GUIDANCE = "framework_guidance"
    BEST_PRACTICES = "best_practices"


class ProgrammingLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    SHELL = "shell"
    OTHER = "other"


class CodingAssistantInput(BaseModel):
    """Input for coding assistant"""
    task_type: CodingTaskType = Field(..., description="Type of coding task")
    programming_language: ProgrammingLanguage = Field(..., description="Primary programming language")
    description: str = Field(..., description="Detailed description of the coding task")
    context: Optional[str] = Field(None, description="Additional context or existing code")
    requirements: List[str] = Field(default_factory=list, description="Specific requirements or constraints")
    frameworks: List[str] = Field(default_factory=list, description="Frameworks or libraries to use")
    search_for_current_info: bool = Field(False, description="Whether to search for current best practices")
    complexity_level: str = Field("intermediate", description="beginner, intermediate, advanced, expert")


class CodeSolution(BaseModel):
    """Generated code solution"""
    code: str = Field(..., description="Generated code")
    language: str = Field(..., description="Programming language")
    explanation: str = Field(..., description="Explanation of the code")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    usage_example: Optional[str] = Field(None, description="Example of how to use the code")
    best_practices: List[str] = Field(default_factory=list, description="Best practices applied")
    potential_improvements: List[str] = Field(default_factory=list, description="Potential improvements")


class CodingAssistantOutput(BaseModel):
    """Output from coding assistant"""
    task_type: str = Field(..., description="Type of task completed")
    solutions: List[CodeSolution] = Field(..., description="Generated code solutions")
    research_summary: Optional[str] = Field(None, description="Summary of web research if performed")
    recommendations: List[str] = Field(default_factory=list, description="Additional recommendations")
    processing_time: float = Field(..., description="Time taken to generate response")


class CodingAssistantTool:
    """MCP tool for coding assistance and code generation"""
    
    def __init__(self, config=None, web_search_tool=None, openrouter_client=None):
        """Initialize with configuration and dependencies"""
        self.config = config or {}
        self.web_search_tool = web_search_tool
        self.openrouter_client = openrouter_client
        self.name = "coding_assistant"
        self.description = "Generate code, provide programming assistance, and offer technical guidance"
        
        # Coding knowledge base
        self.language_specifics = {
            ProgrammingLanguage.PYTHON: {
                "file_extension": ".py",
                "common_frameworks": ["Django", "Flask", "FastAPI", "Pandas", "NumPy"],
                "best_practices": ["PEP 8", "Type hints", "Virtual environments", "Testing with pytest"]
            },
            ProgrammingLanguage.JAVASCRIPT: {
                "file_extension": ".js",
                "common_frameworks": ["React", "Vue", "Angular", "Node.js", "Express"],
                "best_practices": ["ES6+", "Async/await", "Module system", "Testing with Jest"]
            },
            ProgrammingLanguage.TYPESCRIPT: {
                "file_extension": ".ts",
                "common_frameworks": ["React", "Angular", "NestJS", "Next.js"],
                "best_practices": ["Strong typing", "Interfaces", "Generics", "Strict mode"]
            },
            ProgrammingLanguage.JAVA: {
                "file_extension": ".java",
                "common_frameworks": ["Spring", "Maven", "Gradle", "JUnit"],
                "best_practices": ["SOLID principles", "Design patterns", "Unit testing", "Code documentation"]
            },
            ProgrammingLanguage.CSHARP: {
                "file_extension": ".cs",
                "common_frameworks": [".NET", "ASP.NET", "Entity Framework", "NUnit"],
                "best_practices": ["SOLID principles", "Async/await", "LINQ", "Unit testing"]
            }
        }
    
    async def initialize(self):
        """Initialize the coding assistant tool"""
        logger.info("💻 CodingAssistantTool initialized")
    
    async def execute(self, input_data: CodingAssistantInput) -> ToolResponse:
        """Execute coding assistance request"""
        start_time = time.time()
        
        try:
            logger.info(f"💻 Executing coding task: {input_data.task_type.value} in {input_data.programming_language.value}")
            
            # Perform web search if requested
            research_summary = None
            if input_data.search_for_current_info and self.web_search_tool:
                research_summary = await self._perform_web_research(input_data)
            
            # Generate code solutions
            solutions = await self._generate_code_solutions(input_data, research_summary)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(input_data, solutions)
            
            # Create output
            output = CodingAssistantOutput(
                task_type=input_data.task_type.value,
                solutions=solutions,
                research_summary=research_summary,
                recommendations=recommendations,
                processing_time=time.time() - start_time
            )
            
            logger.info(f"✅ Coding assistance completed: {len(solutions)} solutions in {output.processing_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Coding assistance failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="CODING_ASSISTANCE_FAILED",
                    error_message=str(e),
                    details={"task_type": input_data.task_type.value, "language": input_data.programming_language.value}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _perform_web_research(self, input_data: CodingAssistantInput) -> str:
        """Perform web research for current best practices and information"""
        try:
            if not self.web_search_tool:
                return "Web search not available"
            
            # Build search query
            search_terms = [
                input_data.programming_language.value,
                input_data.task_type.value.replace("_", " "),
                "best practices",
                "2024" if input_data.task_type == CodingTaskType.BEST_PRACTICES else ""
            ]
            
            if input_data.frameworks:
                search_terms.extend(input_data.frameworks)
            
            search_query = " ".join(filter(None, search_terms))
            
            # Perform search
            from backend.mcp.tools.web_search_tool import WebSearchInput
            search_input = WebSearchInput(
                query=search_query,
                num_results=20,
                search_type="web"
            )
            
            search_result = await self.web_search_tool.execute(search_input)
            
            if search_result.success:
                # Summarize search results
                results = search_result.data.results
                summary = f"Web research for {input_data.programming_language.value} {input_data.task_type.value}:\n\n"
                
                for result in results[:3]:
                    summary += f"• {result.title}\n  {result.snippet}\n  Source: {result.source}\n\n"
                
                return summary
            else:
                return "Web search failed"
                
        except Exception as e:
            logger.error(f"❌ Web research failed: {e}")
            return f"Web research error: {str(e)}"
    
    async def _generate_code_solutions(
        self, 
        input_data: CodingAssistantInput, 
        research_summary: Optional[str]
    ) -> List[CodeSolution]:
        """Generate code solutions using LLM"""
        try:
            if not self.openrouter_client:
                # Fallback to template-based generation
                return await self._generate_template_solution(input_data)
            
            # Build comprehensive prompt
            prompt = self._build_coding_prompt(input_data, research_summary)
            
            # Get model from settings service
            from services.settings_service import settings_service
            try:
                model = await settings_service.get_llm_model()
                if not model:
                    model = "anthropic/claude-3.5-sonnet"  # Emergency fallback
            except:
                model = "anthropic/claude-3.5-sonnet"  # Emergency fallback
            
            # Import datetime context utility
            from utils.system_prompt_utils import add_datetime_context_to_system_prompt_for_user
            
            # Try to get user_id from config or context if available
            user_id = None
            if self.config and isinstance(self.config, dict):
                user_id = self.config.get('user_id')
            
            system_prompt = await add_datetime_context_to_system_prompt_for_user(
                "You are an expert software engineer with deep knowledge across multiple programming languages and frameworks. Generate high-quality, production-ready code with clear explanations.",
                user_id=user_id
            )
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await self.openrouter_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=3000,
                temperature=0.2  # Lower temperature for more consistent code
            )
            
            # Parse response into solutions
            return await self._parse_llm_response(response.choices[0].message.content, input_data)
            
        except Exception as e:
            logger.error(f"❌ Code generation failed: {e}")
            # Return fallback solution
            return await self._generate_template_solution(input_data)
    
    def _build_coding_prompt(self, input_data: CodingAssistantInput, research_summary: Optional[str]) -> str:
        """Build comprehensive coding prompt"""
        
        language_info = self.language_specifics.get(input_data.programming_language, {})
        
        prompt = f"""
Generate {input_data.programming_language.value} code for the following task:

TASK TYPE: {input_data.task_type.value}
COMPLEXITY LEVEL: {input_data.complexity_level}

DESCRIPTION:
{input_data.description}

REQUIREMENTS:
{chr(10).join(f"• {req}" for req in input_data.requirements) if input_data.requirements else "None specified"}

FRAMEWORKS/LIBRARIES:
{chr(10).join(f"• {fw}" for fw in input_data.frameworks) if input_data.frameworks else "Use standard libraries"}

CONTEXT:
{input_data.context or "No additional context provided"}

LANGUAGE-SPECIFIC BEST PRACTICES:
{chr(10).join(f"• {bp}" for bp in language_info.get('best_practices', [])) if language_info.get('best_practices') else "Apply general best practices"}

{f"CURRENT RESEARCH FINDINGS:{chr(10)}{research_summary}" if research_summary else ""}

Please provide:
1. Complete, working code solution
2. Clear explanation of the approach
3. Dependencies required
4. Usage example
5. Best practices applied
6. Potential improvements

Format your response as:

## Code Solution

```{input_data.programming_language.value}
[Your code here]
```

## Explanation
[Detailed explanation]

## Dependencies
[List of dependencies]

## Usage Example
```{input_data.programming_language.value}
[Usage example]
```

## Best Practices Applied
[List of best practices]

## Potential Improvements
[List of improvements]
"""
        
        return prompt
    
    async def _parse_llm_response(self, response_text: str, input_data: CodingAssistantInput) -> List[CodeSolution]:
        """Parse LLM response into structured code solutions"""
        try:
            # Initialize sections
            sections = {
                "code": "",
                "explanation": "",
                "dependencies": [],
                "usage_example": "",
                "best_practices": [],
                "potential_improvements": []
            }
            
            # Extract code blocks
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', response_text, re.DOTALL)
            if code_blocks:
                sections["code"] = code_blocks[0].strip()
                if len(code_blocks) > 1:
                    sections["usage_example"] = code_blocks[1].strip()
            
            # Extract sections using regex
            section_patterns = {
                "explanation": r'## Explanation\s*\n(.*?)(?=##|$)',
                "dependencies": r'## Dependencies\s*\n(.*?)(?=##|$)',
                "best_practices": r'## Best Practices Applied\s*\n(.*?)(?=##|$)',
                "potential_improvements": r'## Potential Improvements\s*\n(.*?)(?=##|$)'
            }
            
            for section_name, pattern in section_patterns.items():
                match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if section_name in ["dependencies", "best_practices", "potential_improvements"]:
                        # Extract list items
                        items = re.findall(r'[•\-\*]\s*(.+)', content)
                        sections[section_name] = [item.strip() for item in items]
                    else:
                        sections[section_name] = content
            
            # Create solution
            solution = CodeSolution(
                code=sections["code"] or "# Code generation failed - please try again",
                language=input_data.programming_language.value,
                explanation=sections["explanation"] or "Code solution generated based on requirements",
                dependencies=sections["dependencies"],
                usage_example=sections["usage_example"],
                best_practices=sections["best_practices"],
                potential_improvements=sections["potential_improvements"]
            )
            
            return [solution]
            
        except Exception as e:
            logger.error(f"❌ Failed to parse LLM response: {e}")
            return await self._generate_template_solution(input_data)
    
    async def _generate_template_solution(self, input_data: CodingAssistantInput) -> List[CodeSolution]:
        """Generate template-based fallback solution"""
        
        language_info = self.language_specifics.get(input_data.programming_language, {})
        
        if input_data.task_type == CodingTaskType.CODE_GENERATION:
            if input_data.programming_language == ProgrammingLanguage.PYTHON:
                code = f'''# {input_data.description}

def main():
    """
    Main function for: {input_data.description}
    """
    # TODO: Implement your logic here
    print("Hello, World!")
    
    return True

if __name__ == "__main__":
    result = main()
    print(f"Execution completed: {{result}}")'''
            
            elif input_data.programming_language == ProgrammingLanguage.JAVASCRIPT:
                code = f'''// {input_data.description}

function main() {{
    /**
     * Main function for: {input_data.description}
     */
    // TODO: Implement your logic here
    console.log("Hello, World!");
    
    return true;
}}

// Execute if running directly
if (require.main === module) {{
    const result = main();
    console.log(`Execution completed: ${{result}}`);
}}

module.exports = {{ main }};'''
            
            elif input_data.programming_language == ProgrammingLanguage.JAVA:
                code = f'''// {input_data.description}

public class Main {{
    /**
     * Main method for: {input_data.description}
     */
    public static void main(String[] args) {{
        // TODO: Implement your logic here
        System.out.println("Hello, World!");
    }}
}}'''
            
            else:
                code = f"// {input_data.description}\n// TODO: Implement solution"
        
        else:
            code = f"// {input_data.task_type.value} template\n// TODO: Implement {input_data.description}"
        
        solution = CodeSolution(
            code=code,
            language=input_data.programming_language.value,
            explanation=f"Template solution for {input_data.task_type.value} in {input_data.programming_language.value}. This is a basic template that needs to be customized for your specific requirements.",
            dependencies=language_info.get("common_frameworks", [])[:2],
            usage_example=f"# Run the {input_data.programming_language.value} code\n# Follow language-specific execution instructions",
            best_practices=language_info.get("best_practices", [])[:3],
            potential_improvements=[
                "Add comprehensive error handling",
                "Include unit tests",
                "Add detailed documentation",
                "Optimize performance",
                "Add input validation"
            ]
        )
        
        return [solution]
    
    async def _generate_recommendations(
        self, 
        input_data: CodingAssistantInput, 
        solutions: List[CodeSolution]
    ) -> List[str]:
        """Generate additional recommendations"""
        
        recommendations = []
        
        # Language-specific recommendations
        if input_data.programming_language == ProgrammingLanguage.PYTHON:
            recommendations.extend([
                "Consider using virtual environments for dependency management",
                "Add type hints for better code documentation and IDE support",
                "Use pytest for comprehensive testing",
                "Follow PEP 8 style guidelines"
            ])
        
        elif input_data.programming_language == ProgrammingLanguage.JAVASCRIPT:
            recommendations.extend([
                "Consider using TypeScript for better type safety",
                "Implement proper error handling with try-catch blocks",
                "Use modern ES6+ features and async/await",
                "Add ESLint for code quality"
            ])
        
        elif input_data.programming_language == ProgrammingLanguage.JAVA:
            recommendations.extend([
                "Follow SOLID principles for better design",
                "Use Maven or Gradle for dependency management",
                "Implement comprehensive unit tests with JUnit",
                "Add proper logging with SLF4J"
            ])
        
        # Task-specific recommendations
        if input_data.task_type == CodingTaskType.CODE_GENERATION:
            recommendations.extend([
                "Review the generated code for security vulnerabilities",
                "Add comprehensive unit tests",
                "Consider performance implications and optimization",
                "Document the code thoroughly"
            ])
        
        elif input_data.task_type == CodingTaskType.DEBUGGING:
            recommendations.extend([
                "Use debugging tools and breakpoints effectively",
                "Add logging for better error tracking",
                "Consider code review with peers",
                "Write tests to prevent regression"
            ])
        
        elif input_data.task_type == CodingTaskType.ARCHITECTURE_DESIGN:
            recommendations.extend([
                "Consider scalability and maintainability",
                "Apply appropriate design patterns",
                "Plan for testing and deployment",
                "Document architectural decisions"
            ])
        
        # Complexity-specific recommendations
        if input_data.complexity_level in ["advanced", "expert"]:
            recommendations.extend([
                "Consider architectural patterns and design principles",
                "Implement comprehensive error handling and logging",
                "Add performance monitoring and optimization",
                "Plan for scalability and maintainability"
            ])
        
        return recommendations[:6]  # Limit to 6 recommendations
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": CodingAssistantInput.schema(),
            "outputSchema": CodingAssistantOutput.schema()
        }
