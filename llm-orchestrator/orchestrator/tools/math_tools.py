"""
Math Tools - Deterministic mathematical calculations for LangGraph agents
Safe expression evaluation, formula execution, and unit conversions
"""

import logging
import ast
import math
import operator
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


# Allowed AST operations for safe expression evaluation
ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}

# Allowed functions and constants
ALLOWED_FUNCTIONS = {
    "abs": abs,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "max": max,
    "min": min,
    "pi": math.pi,
    "e": math.e
}


class SafeExpressionEvaluator:
    """Safely evaluate mathematical expressions using AST parsing"""
    
    def __init__(self):
        self.variables: Dict[str, float] = {}
    
    def set_variables(self, variables: Dict[str, float]):
        """Set variables for expression evaluation"""
        self.variables = variables or {}
    
    def _eval_node(self, node: ast.AST) -> float:
        """Recursively evaluate AST node"""
        if isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Constant):  # Python >= 3.8
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = ALLOWED_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Operation {type(node.op).__name__} not allowed")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = ALLOWED_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unary operation {type(node.op).__name__} not allowed")
            return op(operand)
        elif isinstance(node, ast.Name):
            if node.id in ALLOWED_FUNCTIONS:
                raise ValueError(f"Function '{node.id}' must be called with parentheses")
            if node.id in self.variables:
                return self.variables[node.id]
            raise ValueError(f"Variable '{node.id}' not defined")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed")
            func_name = node.func.id
            if func_name not in ALLOWED_FUNCTIONS:
                raise ValueError(f"Function '{func_name}' not allowed")
            func = ALLOWED_FUNCTIONS[func_name]
            args = [self._eval_node(arg) for arg in node.args]
            return func(*args)
        else:
            raise ValueError(f"AST node type {type(node).__name__} not allowed")
    
    def evaluate(self, expression: str) -> float:
        """Safely evaluate mathematical expression"""
        try:
            # Parse expression into AST
            tree = ast.parse(expression, mode='eval')
            
            # Evaluate the expression
            result = self._eval_node(tree.body)
            
            return float(result)
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")
        except Exception as e:
            raise ValueError(f"Expression evaluation error: {e}")


def _generate_calculation_steps(expression: str, result: float) -> List[str]:
    """Generate human-readable calculation steps"""
    # Simple step generation - can be enhanced
    steps = [f"Expression: {expression}"]
    steps.append(f"Result: {result}")
    return steps


async def calculate_expression_tool(
    expression: str,
    variables: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Safely evaluate a mathematical expression with optional variable substitution
    
    Args:
        expression: Mathematical expression as string (e.g., "300 * 25 * 1.2")
        variables: Optional dictionary of variable values for substitution
        
    Returns:
        Dictionary with result, steps, and success status
        
    Example:
        result = await calculate_expression_tool("300 * 25 * 1.2")
        # Returns: {"result": 9000.0, "steps": [...], "success": True, "error": None}
    """
    try:
        logger.info(f"Calculating expression: {expression}")
        
        # Create evaluator and set variables
        evaluator = SafeExpressionEvaluator()
        if variables:
            evaluator.set_variables(variables)
        
        # Evaluate expression
        result = evaluator.evaluate(expression)
        
        # Generate steps
        steps = _generate_calculation_steps(expression, result)
        
        logger.info(f"Calculation result: {result}")
        
        return {
            "result": result,
            "steps": steps,
            "expression": expression,
            "variables_used": list(variables.keys()) if variables else [],
            "success": True,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Expression calculation failed: {e}")
        return {
            "result": None,
            "steps": [],
            "expression": expression,
            "variables_used": [],
            "success": False,
            "error": str(e)
        }


async def list_available_formulas_tool() -> Dict[str, Any]:
    """
    List all available formulas in the formula library
    
    Returns:
        Dictionary with formula names, descriptions, and required inputs
    """
    try:
        from orchestrator.tools.math_formulas import FORMULA_LIBRARY
        
        formulas = []
        for formula_name, formula_def in FORMULA_LIBRARY.items():
            formulas.append({
                "name": formula_name,
                "description": formula_def.get("description", ""),
                "required_inputs": formula_def.get("required_inputs", []),
                "optional_inputs": list(formula_def.get("optional_inputs", {}).keys()),
                "output_unit": formula_def.get("output_unit", "")
            })
        
        return {
            "formulas": formulas,
            "count": len(formulas),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Failed to list formulas: {e}")
        return {
            "formulas": [],
            "count": 0,
            "success": False,
            "error": str(e)
        }


# Tool registry (imports done lazily to avoid circular imports)
def _get_math_tools_registry():
    """Get math tools registry with lazy imports"""
    from orchestrator.tools.math_formulas import evaluate_formula_tool
    from orchestrator.tools.unit_conversions import convert_units_tool
    
    return {
        'calculate_expression': calculate_expression_tool,
        'evaluate_formula': evaluate_formula_tool,
        'convert_units': convert_units_tool,
        'list_available_formulas': list_available_formulas_tool
    }

# Create registry
MATH_TOOLS = _get_math_tools_registry()

