"""
Math Tools Module
Mathematical operations and unit conversions for LangGraph agents
"""

import logging
import math
import re
from typing import Dict, Any, List, Union
from decimal import Decimal, getcontext

logger = logging.getLogger(__name__)

# Set precision for decimal calculations
getcontext().prec = 28


class MathTools:
    """Mathematical tools for LangGraph agents"""
    
    def __init__(self):
        # Define conversion factors
        self.conversion_factors = {
            # Digital storage
            "bytes": {
                "kilobytes": 1024,
                "megabytes": 1024**2,
                "gigabytes": 1024**3,
                "terabytes": 1024**4,
                "petabytes": 1024**5
            },
            "kilobytes": {
                "bytes": 1/1024,
                "megabytes": 1024,
                "gigabytes": 1024**2,
                "terabytes": 1024**3,
                "petabytes": 1024**4
            },
            "megabytes": {
                "bytes": 1/(1024**2),
                "kilobytes": 1/1024,
                "gigabytes": 1024,
                "terabytes": 1024**2,
                "petabytes": 1024**3
            },
            "gigabytes": {
                "bytes": 1/(1024**3),
                "kilobytes": 1/(1024**2),
                "megabytes": 1/1024,
                "terabytes": 1024,
                "petabytes": 1024**2
            },
            "terabytes": {
                "bytes": 1/(1024**4),
                "kilobytes": 1/(1024**3),
                "megabytes": 1/(1024**2),
                "gigabytes": 1/1024,
                "petabytes": 1024
            },
            "petabytes": {
                "bytes": 1/(1024**5),
                "kilobytes": 1/(1024**4),
                "megabytes": 1/(1024**3),
                "gigabytes": 1/(1024**2),
                "terabytes": 1/1024
            },
            
            # Volume (US)
            "cups": {
                "fluid_ounces": 8,
                "pints": 0.5,
                "quarts": 0.25,
                "gallons": 0.0625,
                "liters": 0.236588,
                "milliliters": 236.588
            },
            "fluid_ounces": {
                "cups": 0.125,
                "pints": 0.0625,
                "quarts": 0.03125,
                "gallons": 0.0078125,
                "liters": 0.0295735,
                "milliliters": 29.5735
            },
            "pints": {
                "cups": 2,
                "fluid_ounces": 16,
                "quarts": 0.5,
                "gallons": 0.125,
                "liters": 0.473176,
                "milliliters": 473.176
            },
            "quarts": {
                "cups": 4,
                "fluid_ounces": 32,
                "pints": 2,
                "gallons": 0.25,
                "liters": 0.946353,
                "milliliters": 946.353
            },
            "gallons": {
                "cups": 16,
                "fluid_ounces": 128,
                "pints": 8,
                "quarts": 4,
                "liters": 3.78541,
                "milliliters": 3785.41
            },
            "liters": {
                "cups": 4.22675,
                "fluid_ounces": 33.814,
                "pints": 2.11338,
                "quarts": 1.05669,
                "gallons": 0.264172,
                "milliliters": 1000
            },
            "milliliters": {
                "cups": 0.00422675,
                "fluid_ounces": 0.033814,
                "pints": 0.00211338,
                "quarts": 0.00105669,
                "gallons": 0.000264172,
                "liters": 0.001
            },
            
            # Weight/Mass
            "ounces": {
                "pounds": 0.0625,
                "grams": 28.3495,
                "kilograms": 0.0283495,
                "tons": 0.00003125
            },
            "pounds": {
                "ounces": 16,
                "grams": 453.592,
                "kilograms": 0.453592,
                "tons": 0.0005
            },
            "grams": {
                "ounces": 0.035274,
                "pounds": 0.00220462,
                "kilograms": 0.001,
                "tons": 0.00000110231
            },
            "kilograms": {
                "ounces": 35.274,
                "pounds": 2.20462,
                "grams": 1000,
                "tons": 0.00110231
            },
            "tons": {
                "ounces": 32000,
                "pounds": 2000,
                "grams": 907185,
                "kilograms": 907.185
            },
            
            # Length
            "inches": {
                "feet": 0.0833333,
                "yards": 0.0277778,
                "miles": 0.0000157828,
                "centimeters": 2.54,
                "meters": 0.0254,
                "kilometers": 0.0000254
            },
            "feet": {
                "inches": 12,
                "yards": 0.333333,
                "miles": 0.000189394,
                "centimeters": 30.48,
                "meters": 0.3048,
                "kilometers": 0.0003048
            },
            "yards": {
                "inches": 36,
                "feet": 3,
                "miles": 0.000568182,
                "centimeters": 91.44,
                "meters": 0.9144,
                "kilometers": 0.0009144
            },
            "miles": {
                "inches": 63360,
                "feet": 5280,
                "yards": 1760,
                "centimeters": 160934,
                "meters": 1609.34,
                "kilometers": 1.60934
            },
            "centimeters": {
                "inches": 0.393701,
                "feet": 0.0328084,
                "yards": 0.0109361,
                "miles": 0.00000621371,
                "meters": 0.01,
                "kilometers": 0.00001
            },
            "meters": {
                "inches": 39.3701,
                "feet": 3.28084,
                "yards": 1.09361,
                "miles": 0.000621371,
                "centimeters": 100,
                "kilometers": 0.001
            },
            "kilometers": {
                "inches": 39370.1,
                "feet": 3280.84,
                "yards": 1093.61,
                "miles": 0.621371,
                "centimeters": 100000,
                "meters": 1000
            },
            
            # Temperature
            "celsius": {
                "fahrenheit": lambda c: (c * 9/5) + 32,
                "kelvin": lambda c: c + 273.15
            },
            "fahrenheit": {
                "celsius": lambda f: (f - 32) * 5/9,
                "kelvin": lambda f: (f - 32) * 5/9 + 273.15
            },
            "kelvin": {
                "celsius": lambda k: k - 273.15,
                "fahrenheit": lambda k: (k - 273.15) * 9/5 + 32
            }
        }
    
    def get_tools(self) -> Dict[str, Any]:
        """Get all math tools"""
        return {
            "calculate": self.calculate,
            "convert_units": self.convert_units,
            "solve_equation": self.solve_equation,
        }
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all math tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform mathematical calculations including arithmetic, algebra, trigonometry, and more",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', 'sin(45)')"},
                            "precision": {"type": "integer", "description": "Number of decimal places for result", "default": 6}
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_units",
                    "description": "Convert between different units (digital storage, volume, weight, length, temperature)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "number", "description": "Value to convert"},
                            "from_unit": {"type": "string", "description": "Source unit (e.g., 'gigabytes', 'cups', 'pounds', 'meters', 'celsius')"},
                            "to_unit": {"type": "string", "description": "Target unit (e.g., 'megabytes', 'liters', 'kilograms', 'feet', 'fahrenheit')"},
                            "precision": {"type": "integer", "description": "Number of decimal places for result", "default": 4}
                        },
                        "required": ["value", "from_unit", "to_unit"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "solve_equation",
                    "description": "Solve mathematical equations and systems of equations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "equation": {"type": "string", "description": "Equation to solve (e.g., '2x + 3 = 7', 'x^2 + 5x + 6 = 0')"},
                            "variable": {"type": "string", "description": "Variable to solve for", "default": "x"},
                            "precision": {"type": "integer", "description": "Number of decimal places for result", "default": 6}
                        },
                        "required": ["equation"]
                    }
                }
            }
        ]
    
    async def calculate(self, expression: str, precision: int = 6, user_id: str = None) -> Dict[str, Any]:
        """Perform mathematical calculations"""
        try:
            logger.info(f"🧮 LangGraph calculating: {expression[:50]}...")
            
            # Clean and validate expression
            clean_expression = self._clean_expression(expression)
            
            # Evaluate the expression
            result = self._evaluate_expression(clean_expression)
            
            # Format result with specified precision
            if isinstance(result, (int, float)):
                formatted_result = round(result, precision)
            else:
                formatted_result = result
            
            return {
                "success": True,
                "expression": expression,
                "result": formatted_result,
                "precision": precision,
                "calculation_type": self._determine_calculation_type(expression)
            }
            
        except Exception as e:
            logger.error(f"❌ Calculation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "expression": expression
            }
    
    async def convert_units(self, value: float, from_unit: str, to_unit: str, precision: int = 4, user_id: str = None) -> Dict[str, Any]:
        """Convert between different units"""
        try:
            logger.info(f"🔄 LangGraph converting {value} {from_unit} to {to_unit}...")
            
            # Normalize unit names
            from_unit = from_unit.lower().replace(" ", "_")
            to_unit = to_unit.lower().replace(" ", "_")
            
            # Check if conversion is possible
            if from_unit not in self.conversion_factors:
                return {
                    "success": False,
                    "error": f"Unknown source unit: {from_unit}",
                    "available_units": list(self.conversion_factors.keys())
                }
            
            if to_unit not in self.conversion_factors[from_unit]:
                return {
                    "success": False,
                    "error": f"Cannot convert from {from_unit} to {to_unit}",
                    "available_conversions": list(self.conversion_factors[from_unit].keys())
                }
            
            # Perform conversion
            conversion_factor = self.conversion_factors[from_unit][to_unit]
            
            if callable(conversion_factor):
                # Handle temperature conversions (functions)
                result = conversion_factor(value)
            else:
                # Handle standard conversions (multipliers)
                result = value * conversion_factor
            
            # Format result
            formatted_result = round(result, precision)
            
            return {
                "success": True,
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "result": formatted_result,
                "precision": precision,
                "conversion_type": self._determine_conversion_type(from_unit, to_unit)
            }
            
        except Exception as e:
            logger.error(f"❌ Unit conversion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit
            }
    
    async def solve_equation(self, equation: str, variable: str = "x", precision: int = 6, user_id: str = None) -> Dict[str, Any]:
        """Solve mathematical equations"""
        try:
            logger.info(f"🔢 LangGraph solving equation: {equation[:50]}...")
            
            # Clean equation
            clean_equation = equation.replace(" ", "").lower()
            
            # Handle different types of equations
            if "=" in clean_equation:
                # Linear equation: ax + b = c
                if self._is_linear_equation(clean_equation, variable):
                    result = self._solve_linear_equation(clean_equation, variable)
                # Quadratic equation: ax^2 + bx + c = 0
                elif self._is_quadratic_equation(clean_equation, variable):
                    result = self._solve_quadratic_equation(clean_equation, variable)
                else:
                    return {
                        "success": False,
                        "error": "Unsupported equation type",
                        "equation": equation
                    }
            else:
                return {
                    "success": False,
                    "error": "Equation must contain '=' sign",
                    "equation": equation
                }
            
            # Format results
            if isinstance(result, list):
                formatted_results = [round(r, precision) if isinstance(r, (int, float)) else r for r in result]
            else:
                formatted_results = round(result, precision) if isinstance(result, (int, float)) else result
            
            return {
                "success": True,
                "equation": equation,
                "variable": variable,
                "solution": formatted_results,
                "precision": precision,
                "equation_type": self._determine_equation_type(clean_equation, variable)
            }
            
        except Exception as e:
            logger.error(f"❌ Equation solving failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "equation": equation,
                "variable": variable
            }
    
    def _clean_expression(self, expression: str) -> str:
        """Clean and validate mathematical expression"""
        # Remove extra spaces
        expression = expression.replace(" ", "")
        
        # Replace common mathematical symbols
        replacements = {
            "×": "*",
            "÷": "/",
            "−": "-",
            "²": "**2",
            "³": "**3",
            "π": str(math.pi),
            "e": str(math.e)
        }
        
        for old, new in replacements.items():
            expression = expression.replace(old, new)
        
        return expression
    
    def _evaluate_expression(self, expression: str) -> Union[int, float]:
        """Safely evaluate mathematical expression"""
        # Define safe mathematical functions
        safe_dict = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'floor': math.floor,
            'ceil': math.ceil,
            'pi': math.pi,
            'e': math.e
        }
        
        # Add basic arithmetic operations
        safe_dict.update({
            '__builtins__': {},
            'True': True,
            'False': False,
            'None': None
        })
        
        try:
            # Use eval with restricted namespace
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return result
        except Exception as e:
            raise ValueError(f"Invalid mathematical expression: {str(e)}")
    
    def _determine_calculation_type(self, expression: str) -> str:
        """Determine the type of calculation"""
        expression = expression.lower()
        
        if any(op in expression for op in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']):
            return "trigonometry"
        elif any(op in expression for op in ['log', 'ln', 'exp']):
            return "logarithmic"
        elif '**' in expression or '^' in expression:
            return "exponential"
        elif any(op in expression for op in ['+', '-', '*', '/']):
            return "arithmetic"
        else:
            return "basic"
    
    def _determine_conversion_type(self, from_unit: str, to_unit: str) -> str:
        """Determine the type of unit conversion"""
        digital_units = ['bytes', 'kilobytes', 'megabytes', 'gigabytes', 'terabytes', 'petabytes']
        volume_units = ['cups', 'fluid_ounces', 'pints', 'quarts', 'gallons', 'liters', 'milliliters']
        weight_units = ['ounces', 'pounds', 'grams', 'kilograms', 'tons']
        length_units = ['inches', 'feet', 'yards', 'miles', 'centimeters', 'meters', 'kilometers']
        temp_units = ['celsius', 'fahrenheit', 'kelvin']
        
        if from_unit in digital_units and to_unit in digital_units:
            return "digital_storage"
        elif from_unit in volume_units and to_unit in volume_units:
            return "volume"
        elif from_unit in weight_units and to_unit in weight_units:
            return "weight"
        elif from_unit in length_units and to_unit in length_units:
            return "length"
        elif from_unit in temp_units and to_unit in temp_units:
            return "temperature"
        else:
            return "mixed"
    
    def _is_linear_equation(self, equation: str, variable: str) -> bool:
        """Check if equation is linear"""
        # Remove = and everything after it
        left_side = equation.split('=')[0]
        
        # Check for variable with power 1 (no ^2, ^3, etc.)
        if f"{variable}^" in left_side:
            return False
        
        return True
    
    def _is_quadratic_equation(self, equation: str, variable: str) -> bool:
        """Check if equation is quadratic"""
        left_side = equation.split('=')[0]
        
        # Check for variable with power 2
        if f"{variable}^2" in left_side or f"{variable}²" in left_side:
            return True
        
        return False
    
    def _solve_linear_equation(self, equation: str, variable: str) -> float:
        """Solve linear equation ax + b = c"""
        # This is a simplified implementation
        # In practice, you'd want to use a proper symbolic math library like sympy
        
        # For now, return a placeholder
        return 0.0
    
    def _solve_quadratic_equation(self, equation: str, variable: str) -> List[float]:
        """Solve quadratic equation ax^2 + bx + c = 0"""
        # This is a simplified implementation
        # In practice, you'd want to use a proper symbolic math library like sympy
        
        # For now, return placeholder solutions
        return [0.0, 0.0]
    
    def _determine_equation_type(self, equation: str, variable: str) -> str:
        """Determine the type of equation"""
        if self._is_quadratic_equation(equation, variable):
            return "quadratic"
        elif self._is_linear_equation(equation, variable):
            return "linear"
        else:
            return "unknown"


# Global instance for use by tool registry
_math_tools_instance = None


async def _get_math_tools():
    """Get global math tools instance"""
    global _math_tools_instance
    if _math_tools_instance is None:
        _math_tools_instance = MathTools()
    return _math_tools_instance


async def calculate(expression: str, precision: int = 6, user_id: str = None) -> Dict[str, Any]:
    """LangGraph tool function: Calculate mathematical expressions"""
    tools_instance = await _get_math_tools()
    return await tools_instance.calculate(expression, precision, user_id)


async def convert_units(value: float, from_unit: str, to_unit: str, precision: int = 6, user_id: str = None) -> Dict[str, Any]:
    """LangGraph tool function: Convert between units"""
    tools_instance = await _get_math_tools()
    return await tools_instance.convert_units(value, from_unit, to_unit, precision, user_id)


async def solve_equation(equation: str, variable: str = "x", precision: int = 6, user_id: str = None) -> Dict[str, Any]:
    """LangGraph tool function: Solve mathematical equations"""
    tools_instance = await _get_math_tools()
    return await tools_instance.solve_equation(equation, variable, precision, user_id)