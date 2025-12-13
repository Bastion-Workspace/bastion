"""
Math Formulas - Pre-defined formula library for common calculations
HVAC/BTU, electrical, construction, and general engineering formulas
"""

import logging
import math
from typing import Dict, Any, Callable, List

logger = logging.getLogger(__name__)


# Formula library structure
FORMULA_LIBRARY: Dict[str, Dict[str, Any]] = {
    # HVAC/BTU Calculations
    "btu_hvac": {
        "description": "Calculate BTU requirements for HVAC room sizing",
        "formula": lambda inputs: (
            inputs["square_feet"] * 
            inputs.get("btu_per_sqft", 25) * 
            inputs.get("climate_factor", 1.0) *
            inputs.get("ceiling_height_factor", 1.0) *
            inputs.get("insulation_factor", 1.0)
        ),
        "required_inputs": ["square_feet"],
        "optional_inputs": {
            "btu_per_sqft": 25,  # Default for moderate climate
            "climate_factor": 1.0,  # 1.0 = moderate, 1.2-1.5 = hot, 0.8-0.9 = cold
            "ceiling_height_factor": 1.0,  # 1.0 = 8ft, adjust for taller ceilings
            "insulation_factor": 1.0  # 1.0 = standard, <1.0 = well insulated, >1.0 = poor insulation
        },
        "output_unit": "BTU/hr",
        "validation": lambda inputs: inputs["square_feet"] > 0,
        "step_generator": lambda inputs, result: [
            f"Base calculation: {inputs['square_feet']} sq ft × {inputs.get('btu_per_sqft', 25)} BTU/sq ft = {inputs['square_feet'] * inputs.get('btu_per_sqft', 25)} BTU/hr",
            f"Apply climate factor: × {inputs.get('climate_factor', 1.0)} = {inputs['square_feet'] * inputs.get('btu_per_sqft', 25) * inputs.get('climate_factor', 1.0)} BTU/hr",
            f"Apply ceiling height factor: × {inputs.get('ceiling_height_factor', 1.0)} = {inputs['square_feet'] * inputs.get('btu_per_sqft', 25) * inputs.get('climate_factor', 1.0) * inputs.get('ceiling_height_factor', 1.0)} BTU/hr",
            f"Apply insulation factor: × {inputs.get('insulation_factor', 1.0)} = {result} BTU/hr"
        ]
    },
    
    # Electrical Calculations
    "ohms_law_voltage": {
        "description": "Calculate voltage using Ohm's Law (V = I × R)",
        "formula": lambda inputs: inputs["current"] * inputs["resistance"],
        "required_inputs": ["current", "resistance"],
        "optional_inputs": {},
        "output_unit": "volts",
        "validation": lambda inputs: inputs["current"] >= 0 and inputs["resistance"] >= 0,
        "step_generator": lambda inputs, result: [
            f"Voltage = Current × Resistance",
            f"V = {inputs['current']} A × {inputs['resistance']} Ω = {result} V"
        ]
    },
    "ohms_law_current": {
        "description": "Calculate current using Ohm's Law (I = V / R)",
        "formula": lambda inputs: inputs["voltage"] / inputs["resistance"],
        "required_inputs": ["voltage", "resistance"],
        "optional_inputs": {},
        "output_unit": "amps",
        "validation": lambda inputs: inputs["voltage"] >= 0 and inputs["resistance"] > 0,
        "step_generator": lambda inputs, result: [
            f"Current = Voltage / Resistance",
            f"I = {inputs['voltage']} V / {inputs['resistance']} Ω = {result} A"
        ]
    },
    "ohms_law_resistance": {
        "description": "Calculate resistance using Ohm's Law (R = V / I)",
        "formula": lambda inputs: inputs["voltage"] / inputs["current"],
        "required_inputs": ["voltage", "current"],
        "optional_inputs": {},
        "output_unit": "ohms",
        "validation": lambda inputs: inputs["voltage"] >= 0 and inputs["current"] > 0,
        "step_generator": lambda inputs, result: [
            f"Resistance = Voltage / Current",
            f"R = {inputs['voltage']} V / {inputs['current']} A = {result} Ω"
        ]
    },
    "power_dissipation": {
        "description": "Calculate power dissipation (P = V² / R or P = I² × R)",
        "formula": lambda inputs: (
            inputs.get("voltage", 0) ** 2 / inputs.get("resistance", 1)
            if inputs.get("voltage") is not None and inputs.get("resistance") is not None
            else inputs.get("current", 0) ** 2 * inputs.get("resistance", 0)
        ),
        "required_inputs": [],  # Either (voltage, resistance) or (current, resistance)
        "optional_inputs": {
            "voltage": None,
            "current": None,
            "resistance": None
        },
        "output_unit": "watts",
        "validation": lambda inputs: (
            (inputs.get("voltage") is not None and inputs.get("resistance") is not None and inputs.get("resistance") > 0) or
            (inputs.get("current") is not None and inputs.get("resistance") is not None and inputs.get("resistance") >= 0)
        ),
        "step_generator": lambda inputs, result: [
            f"Power = {'V² / R' if inputs.get('voltage') else 'I² × R'}",
            f"P = {inputs.get('voltage', inputs.get('current', 0))}² / {inputs.get('resistance', 1)} = {result} W"
        ] if inputs.get("voltage") else [
            f"Power = I² × R",
            f"P = {inputs.get('current', 0)}² × {inputs.get('resistance', 0)} = {result} W"
        ]
    },
    "voltage_divider": {
        "description": "Calculate output voltage of a resistor voltage divider (Vout = Vin × R2 / (R1 + R2))",
        "formula": lambda inputs: inputs["input_voltage"] * inputs["r2"] / (inputs["r1"] + inputs["r2"]),
        "required_inputs": ["input_voltage", "r1", "r2"],
        "optional_inputs": {},
        "output_unit": "volts",
        "validation": lambda inputs: inputs["input_voltage"] >= 0 and inputs["r1"] > 0 and inputs["r2"] > 0,
        "step_generator": lambda inputs, result: [
            f"Vout = Vin × R2 / (R1 + R2)",
            f"Vout = {inputs['input_voltage']} V × {inputs['r2']} Ω / ({inputs['r1']} Ω + {inputs['r2']} Ω)",
            f"Vout = {inputs['input_voltage']} V × {inputs['r2']} Ω / {inputs['r1'] + inputs['r2']} Ω = {result} V"
        ]
    },
    "capacitor_impedance": {
        "description": "Calculate capacitor impedance (Xc = 1 / (2πfC))",
        "formula": lambda inputs: 1 / (2 * math.pi * inputs["frequency"] * inputs["capacitance"]),
        "required_inputs": ["frequency", "capacitance"],
        "optional_inputs": {},
        "output_unit": "ohms",
        "validation": lambda inputs: inputs["frequency"] > 0 and inputs["capacitance"] > 0,
        "step_generator": lambda inputs, result: [
            f"Capacitive Reactance Xc = 1 / (2π × f × C)",
            f"Xc = 1 / (2π × {inputs['frequency']} Hz × {inputs['capacitance']} F) = {result} Ω"
        ]
    },
    
    # Construction/General
    "area_rectangle": {
        "description": "Calculate area of a rectangle (Area = length × width)",
        "formula": lambda inputs: inputs["length"] * inputs["width"],
        "required_inputs": ["length", "width"],
        "optional_inputs": {},
        "output_unit": "square units",
        "validation": lambda inputs: inputs["length"] > 0 and inputs["width"] > 0,
        "step_generator": lambda inputs, result: [
            f"Area = Length × Width",
            f"Area = {inputs['length']} × {inputs['width']} = {result}"
        ]
    },
    "volume_rectangular": {
        "description": "Calculate volume of a rectangular prism (Volume = length × width × height)",
        "formula": lambda inputs: inputs["length"] * inputs["width"] * inputs["height"],
        "required_inputs": ["length", "width", "height"],
        "optional_inputs": {},
        "output_unit": "cubic units",
        "validation": lambda inputs: inputs["length"] > 0 and inputs["width"] > 0 and inputs["height"] > 0,
        "step_generator": lambda inputs, result: [
            f"Volume = Length × Width × Height",
            f"Volume = {inputs['length']} × {inputs['width']} × {inputs['height']} = {result}"
        ]
    },
    "material_quantity": {
        "description": "Estimate material quantity based on area and coverage rate",
        "formula": lambda inputs: inputs["area"] / inputs.get("coverage_per_unit", 1),
        "required_inputs": ["area"],
        "optional_inputs": {
            "coverage_per_unit": 1  # e.g., sq ft per gallon of paint
        },
        "output_unit": "units",
        "validation": lambda inputs: inputs["area"] > 0 and inputs.get("coverage_per_unit", 1) > 0,
        "step_generator": lambda inputs, result: [
            f"Quantity = Area / Coverage per unit",
            f"Quantity = {inputs['area']} / {inputs.get('coverage_per_unit', 1)} = {result}"
        ]
    }
}


async def evaluate_formula_tool(
    formula_name: str,
    inputs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a pre-defined formula from the formula library
    
    Args:
        formula_name: Name of formula (e.g., "btu_hvac", "ohms_law_voltage")
        inputs: Dictionary of input values for the formula
        
    Returns:
        Dictionary with result, steps, formula used, and success status
        
    Example:
        result = await evaluate_formula_tool(
            formula_name="btu_hvac",
            inputs={"square_feet": 300, "climate_factor": 1.2}
        )
        # Returns: {"result": 9000.0, "unit": "BTU/hr", "formula_used": "btu_hvac", "steps": [...], "success": True}
    """
    try:
        logger.info(f"Evaluating formula: {formula_name} with inputs: {inputs}")
        
        if formula_name not in FORMULA_LIBRARY:
            return {
                "result": None,
                "unit": None,
                "formula_used": formula_name,
                "steps": [],
                "success": False,
                "error": f"Formula '{formula_name}' not found in library"
            }
        
        formula_def = FORMULA_LIBRARY[formula_name]
        
        # Validate required inputs
        required_inputs = formula_def.get("required_inputs", [])
        missing_inputs = [inp for inp in required_inputs if inp not in inputs]
        if missing_inputs:
            return {
                "result": None,
                "unit": None,
                "formula_used": formula_name,
                "steps": [],
                "success": False,
                "error": f"Missing required inputs: {', '.join(missing_inputs)}"
            }
        
        # Fill in optional inputs with defaults
        optional_inputs = formula_def.get("optional_inputs", {})
        for opt_key, opt_default in optional_inputs.items():
            if opt_key not in inputs:
                inputs[opt_key] = opt_default
        
        # Validate inputs
        validation_func = formula_def.get("validation")
        if validation_func:
            try:
                if not validation_func(inputs):
                    return {
                        "result": None,
                        "unit": None,
                        "formula_used": formula_name,
                        "steps": [],
                        "success": False,
                        "error": f"Input validation failed for formula '{formula_name}'"
                    }
            except Exception as e:
                return {
                    "result": None,
                    "unit": None,
                    "formula_used": formula_name,
                    "steps": [],
                    "success": False,
                    "error": f"Validation error: {str(e)}"
                }
        
        # Evaluate formula
        formula_func = formula_def["formula"]
        result = formula_func(inputs)
        
        # Generate steps if step generator available
        steps = []
        step_generator = formula_def.get("step_generator")
        if step_generator:
            try:
                steps = step_generator(inputs, result)
            except Exception as e:
                logger.warning(f"Step generation failed: {e}")
                steps = [f"Formula: {formula_name}", f"Result: {result}"]
        else:
            steps = [f"Formula: {formula_name}", f"Result: {result}"]
        
        output_unit = formula_def.get("output_unit", "")
        
        logger.info(f"Formula evaluation result: {result} {output_unit}")
        
        return {
            "result": float(result),
            "unit": output_unit,
            "formula_used": formula_name,
            "formula_description": formula_def.get("description", ""),
            "steps": steps,
            "inputs_used": inputs,
            "success": True,
            "error": None
        }
        
    except ZeroDivisionError:
        return {
            "result": None,
            "unit": None,
            "formula_used": formula_name,
            "steps": [],
            "success": False,
            "error": "Division by zero in formula calculation"
        }
    except Exception as e:
        logger.error(f"Formula evaluation failed: {e}")
        return {
            "result": None,
            "unit": None,
            "formula_used": formula_name,
            "steps": [],
            "success": False,
            "error": str(e)
        }

