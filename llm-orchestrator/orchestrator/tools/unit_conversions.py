"""
Unit Conversions - Comprehensive unit conversion system for math tools
Supports length, area, volume, temperature, power, and electrical units
"""

import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


# Unit conversion tables
UNIT_CONVERSIONS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "length": {
        "inches": {
            "feet": 1/12,
            "meters": 0.0254,
            "centimeters": 2.54,
            "millimeters": 25.4
        },
        "feet": {
            "inches": 12,
            "meters": 0.3048,
            "centimeters": 30.48,
            "millimeters": 304.8
        },
        "meters": {
            "inches": 39.3701,
            "feet": 3.28084,
            "centimeters": 100,
            "millimeters": 1000
        },
        "centimeters": {
            "inches": 0.393701,
            "feet": 0.0328084,
            "meters": 0.01,
            "millimeters": 10
        },
        "millimeters": {
            "inches": 0.0393701,
            "feet": 0.00328084,
            "meters": 0.001,
            "centimeters": 0.1
        }
    },
    "area": {
        "sq_ft": {
            "sq_m": 0.092903,
            "sq_in": 144,
            "acres": 0.0000229568
        },
        "sq_m": {
            "sq_ft": 10.7639,
            "sq_in": 1550.0031,
            "acres": 0.000247105
        },
        "sq_in": {
            "sq_ft": 1/144,
            "sq_m": 0.00064516,
            "acres": 0.000000159423
        },
        "acres": {
            "sq_ft": 43560,
            "sq_m": 4046.86,
            "sq_in": 6272640
        }
    },
    "volume": {
        "cubic_ft": {
            "cubic_m": 0.0283168,
            "liters": 28.3168,
            "gallons": 7.48052
        },
        "cubic_m": {
            "cubic_ft": 35.3147,
            "liters": 1000,
            "gallons": 264.172
        },
        "liters": {
            "cubic_ft": 0.0353147,
            "cubic_m": 0.001,
            "gallons": 0.264172
        },
        "gallons": {
            "cubic_ft": 0.133681,
            "cubic_m": 0.00378541,
            "liters": 3.78541
        }
    },
    "power": {
        "watts": {
            "kilowatts": 0.001,
            "btu_per_hr": 3.41214,
            "horsepower": 0.00134102
        },
        "kilowatts": {
            "watts": 1000,
            "btu_per_hr": 3412.14,
            "horsepower": 1.34102
        },
        "btu_per_hr": {
            "watts": 0.293071,
            "kilowatts": 0.000293071,
            "horsepower": 0.000393014
        },
        "horsepower": {
            "watts": 745.7,
            "kilowatts": 0.7457,
            "btu_per_hr": 2544.43
        }
    },
    "electrical": {
        "volts": {
            "millivolts": 1000
        },
        "millivolts": {
            "volts": 0.001
        },
        "amps": {
            "milliamps": 1000
        },
        "milliamps": {
            "amps": 0.001
        },
        "ohms": {
            "kilohms": 0.001,
            "megohms": 0.000001
        },
        "kilohms": {
            "ohms": 1000,
            "megohms": 0.001
        },
        "megohms": {
            "ohms": 1000000,
            "kilohms": 1000
        }
    }
}

# Temperature conversions (non-linear, require functions)
TEMPERATURE_CONVERSIONS: Dict[str, Dict[str, Callable]] = {
    "fahrenheit": {
        "celsius": lambda f: (f - 32) * 5/9,
        "kelvin": lambda f: (f - 32) * 5/9 + 273.15
    },
    "celsius": {
        "fahrenheit": lambda c: c * 9/5 + 32,
        "kelvin": lambda c: c + 273.15
    },
    "kelvin": {
        "fahrenheit": lambda k: (k - 273.15) * 9/5 + 32,
        "celsius": lambda k: k - 273.15
    }
}


def _detect_quantity_type(from_unit: str, to_unit: str) -> Optional[str]:
    """Detect quantity type from unit names"""
    unit_lower = from_unit.lower()
    
    # Check each category
    for category, units in UNIT_CONVERSIONS.items():
        if unit_lower in units:
            # Verify to_unit is also in same category
            if to_unit.lower() in units or to_unit.lower() in units.get(unit_lower, {}):
                return category
    
    # Check temperature
    if unit_lower in TEMPERATURE_CONVERSIONS:
        if to_unit.lower() in TEMPERATURE_CONVERSIONS[unit_lower]:
            return "temperature"
    
    return None


async def convert_units_tool(
    value: float,
    from_unit: str,
    to_unit: str,
    quantity_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert between units with automatic quantity type detection
    
    Args:
        value: Numeric value to convert
        from_unit: Source unit (e.g., "sq_ft", "fahrenheit", "watts")
        to_unit: Target unit (e.g., "sq_m", "celsius", "kilowatts")
        quantity_type: Optional quantity type hint ("length", "area", "volume", "temperature", "power", "electrical")
        
    Returns:
        Dictionary with converted value, conversion factor, and success status
        
    Example:
        result = await convert_units_tool(300, "sq_ft", "sq_m", "area")
        # Returns: {"result": 27.87, "from_unit": "sq_ft", "to_unit": "sq_m", "conversion_factor": 0.092903, "success": True}
    """
    try:
        logger.info(f"Converting {value} {from_unit} to {to_unit}")
        
        # Auto-detect quantity type if not provided
        if not quantity_type:
            quantity_type = _detect_quantity_type(from_unit, to_unit)
        
        if not quantity_type:
            return {
                "result": None,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "conversion_factor": None,
                "success": False,
                "error": f"Could not determine quantity type for units '{from_unit}' and '{to_unit}'"
            }
        
        # Handle temperature conversions (non-linear)
        if quantity_type == "temperature":
            from_unit_lower = from_unit.lower()
            to_unit_lower = to_unit.lower()
            
            if from_unit_lower not in TEMPERATURE_CONVERSIONS:
                return {
                    "result": None,
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "conversion_factor": None,
                    "success": False,
                    "error": f"Unknown temperature unit: {from_unit}"
                }
            
            if to_unit_lower not in TEMPERATURE_CONVERSIONS[from_unit_lower]:
                return {
                    "result": None,
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "conversion_factor": None,
                    "success": False,
                    "error": f"Cannot convert from {from_unit} to {to_unit}"
                }
            
            conversion_func = TEMPERATURE_CONVERSIONS[from_unit_lower][to_unit_lower]
            result = conversion_func(value)
            
            return {
                "result": result,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "conversion_factor": None,  # Non-linear, no single factor
                "conversion_type": "non_linear",
                "success": True,
                "error": None
            }
        
        # Handle linear conversions
        if quantity_type not in UNIT_CONVERSIONS:
            return {
                "result": None,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "conversion_factor": None,
                "success": False,
                "error": f"Unknown quantity type: {quantity_type}"
            }
        
        category_units = UNIT_CONVERSIONS[quantity_type]
        from_unit_lower = from_unit.lower()
        to_unit_lower = to_unit.lower()
        
        if from_unit_lower not in category_units:
            return {
                "result": None,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "conversion_factor": None,
                "success": False,
                "error": f"Unknown {quantity_type} unit: {from_unit}"
            }
        
        from_unit_conversions = category_units[from_unit_lower]
        
        if to_unit_lower not in from_unit_conversions:
            return {
                "result": None,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "conversion_factor": None,
                "success": False,
                "error": f"Cannot convert from {from_unit} to {to_unit} in {quantity_type} category"
            }
        
        conversion_factor = from_unit_conversions[to_unit_lower]
        result = value * conversion_factor
        
        logger.info(f"Conversion result: {result} {to_unit}")
        
        return {
            "result": result,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "conversion_factor": conversion_factor,
            "conversion_type": "linear",
            "success": True,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Unit conversion failed: {e}")
        return {
            "result": None,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "conversion_factor": None,
            "success": False,
            "error": str(e)
        }

