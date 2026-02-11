"""
Math Formulas - Pre-defined formula library for common calculations
HVAC/BTU, electrical, construction, and general engineering formulas
"""

import logging
import math
from typing import Dict, Any, Callable, List

logger = logging.getLogger(__name__)


def _calculate_manual_j_heat_loss(inputs: Dict[str, Any]) -> float:
    """
    Calculate heat loss according to Manual J methodology
    
    Formula components:
    1. Conduction losses: Q = U × A × ΔT
    2. Infiltration losses: Q = 0.018 × V × ACH × ΔT × (1 - ERV_efficiency)
    3. Ventilation losses: Q = 1.08 × CFM × ΔT × (1 - HRV_efficiency)
    4. Internal gains: Subtracted from total losses
    
    Where:
    - U = 1/R (thermal transmittance)
    - A = area in sq ft
    - ΔT = temperature difference (indoor - outdoor)
    - V = volume in cubic feet
    - ACH = air changes per hour
    - CFM = cubic feet per minute
    
    Can calculate:
    - Whole-building heat loss (pass floor_area or rooms list)
    - Per-room heat loss (set per_room_calculation=True with rooms list)
    """
    # Check if this is a per-room calculation request
    per_room_calc = inputs.get("per_room_calculation", False)
    rooms = inputs.get("rooms") or []
    
    # If per-room calculation requested, process each room individually
    if per_room_calc and isinstance(rooms, list) and len(rooms) > 0:
        results = []
        total_building_loss = 0.0
        
        for room in rooms:
            room_name = room.get("name", "Unnamed Room")
            room_area = _room_area_sqft(room)
            
            if room_area == 0:
                continue
            
            # Create inputs for this specific room
            room_inputs = inputs.copy()
            room_inputs["floor_area"] = room_area
            room_inputs["rooms"] = None  # Prevent recursion
            room_inputs["per_room_calculation"] = False  # Calculate this room as a "building"
            
            # Proportionally allocate whole-building envelope to this room by floor area.
            # Otherwise we'd apply the full building's windows/doors/walls to every room,
            # which wildly overstates heat loss for small rooms (e.g. 48 sq ft bathroom
            # with 275 sq ft of windows).
            total_area = sum(_room_area_sqft(r) for r in rooms)
            room_proportion = room_area / total_area if total_area > 0 else 0
            
            room_inputs["wall_area"] = (room_proportion * (inputs.get("wall_area") or 0)) or 0.0
            room_inputs["window_area"] = (room_proportion * (inputs.get("window_area") or 0)) or 0.0
            room_inputs["door_area"] = (room_proportion * (inputs.get("door_area") or 0)) or 0.0
            # roof_area is handled inside _calculate_manual_j_single_space (defaults to floor_area)
            # ventilation_cfm: allocate by proportion so total CFM is preserved conceptually
            room_inputs["ventilation_cfm"] = room_proportion * (inputs.get("ventilation_cfm") or 0.0)
            
            # Calculate heat loss for this room as if it were its own building
            room_loss = _calculate_manual_j_single_space(room_inputs)
            
            total_building_loss += room_loss
            results.append({
                "room_name": room_name,
                "floor_area": room_area,
                "heat_loss_btu_hr": round(room_loss, 1),
                "proportion_of_total": round(room_proportion * 100, 1)
            })
        
        # Store per-room results in inputs for step generator
        inputs["_per_room_results"] = results
        inputs["_total_building_loss"] = total_building_loss
        
        return total_building_loss
    
    # Standard whole-building calculation
    return _calculate_manual_j_single_space(inputs)


def _calculate_manual_j_single_space(inputs: Dict[str, Any]) -> float:
    """
    Calculate Manual J heat loss for a single space (room or whole building).
    This is the core calculation logic.
    """
    # Temperature difference
    delta_t = inputs["indoor_design_temp"] - inputs["outdoor_design_temp"]
    
    # Building/room volume - accept floor_area OR sum rooms
    floor_area = inputs.get("floor_area")
    if floor_area is None or floor_area == 0:
        # Sum room areas if provided (for whole-building mode)
        rooms = inputs.get("rooms") or []
        floor_area = sum(_room_area_sqft(r) for r in rooms) if isinstance(rooms, list) else 0.0
    if floor_area == 0:
        raise ValueError("floor_area is required (or rooms list with areas)")
    
    ceiling_height = inputs.get("ceiling_height", 8.0)
    volume = floor_area * ceiling_height
    
    total_loss = 0.0
    loss_components = {}
    
    # 1. Wall conduction losses
    wall_area = inputs.get("wall_area", 0.0)
    if wall_area > 0:
        wall_r = inputs.get("wall_r_value", 13.0)
        wall_u = 1.0 / wall_r if wall_r > 0 else 0
        wall_loss = wall_u * wall_area * delta_t
        total_loss += wall_loss
        loss_components["wall_loss"] = wall_loss
    
    # 2. Roof/Ceiling conduction losses
    roof_area = inputs.get("roof_area", 0.0)
    if roof_area == 0:
        roof_area = floor_area  # Default to floor area if not specified
    roof_r = inputs.get("roof_r_value", 30.0)
    roof_u = 1.0 / roof_r if roof_r > 0 else 0
    roof_loss = roof_u * roof_area * delta_t
    total_loss += roof_loss
    loss_components["roof_loss"] = roof_loss
    
    # 3. Floor conduction losses
    floor_over_unconditioned = inputs.get("floor_over_unconditioned", False)
    if floor_over_unconditioned:
        floor_r = inputs.get("floor_r_value", 19.0)
        floor_u = 1.0 / floor_r if floor_r > 0 else 0
        floor_loss = floor_u * floor_area * delta_t
        total_loss += floor_loss
        loss_components["floor_loss"] = floor_loss
    
    # 4. Window conduction losses
    window_area = inputs.get("window_area", 0.0)
    if window_area > 0:
        window_u = inputs.get("window_u_value", 0.5)
        window_loss = window_u * window_area * delta_t
        total_loss += window_loss
        loss_components["window_loss"] = window_loss
    
    # 5. Door conduction losses
    door_area = inputs.get("door_area", 0.0)
    if door_area > 0:
        door_u = inputs.get("door_u_value", 0.2)
        door_loss = door_u * door_area * delta_t
        total_loss += door_loss
        loss_components["door_loss"] = door_loss
    
    # 6. Infiltration losses
    # Q = 0.018 × V × ACH × ΔT (BTU/hr)
    # 0.018 = air density × specific heat conversion factor
    ach = inputs.get("air_changes_per_hour", 0.5)
    infiltration_loss = 0.018 * volume * ach * delta_t
    total_loss += infiltration_loss
    loss_components["infiltration_loss"] = infiltration_loss
    
    # 7. Ventilation losses
    # Q = 1.08 × CFM × ΔT (BTU/hr)
    # 1.08 = air density × specific heat × 60 min/hr conversion
    ventilation_cfm = inputs.get("ventilation_cfm", 0.0)
    if ventilation_cfm > 0:
        ventilation_loss = 1.08 * ventilation_cfm * delta_t
        total_loss += ventilation_loss
        loss_components["ventilation_loss"] = ventilation_loss
    
    # 8. Internal heat gains (subtract from losses)
    internal_gains = 0.0
    
    # Occupant heat gain: ~400 BTU/hr per person
    occupant_count = inputs.get("occupant_count", 0)
    if occupant_count > 0:
        occupant_gain = occupant_count * 400
        internal_gains += occupant_gain
        loss_components["occupant_gain"] = occupant_gain
    
    # Appliance heat gain
    appliance_gain = inputs.get("appliance_heat_gain", 0.0)
    if appliance_gain > 0:
        internal_gains += appliance_gain
        loss_components["appliance_gain"] = appliance_gain
    
    # Lighting heat gain
    lighting_gain = inputs.get("lighting_heat_gain", 0.0)
    if lighting_gain > 0:
        internal_gains += lighting_gain
        loss_components["lighting_gain"] = lighting_gain
    
    # Net heat loss (losses minus gains)
    net_heat_loss = max(0.0, total_loss - internal_gains)
    loss_components["total_conduction_loss"] = total_loss - infiltration_loss - (loss_components.get("ventilation_loss", 0.0))
    loss_components["total_loss_before_gains"] = total_loss
    loss_components["total_internal_gains"] = internal_gains
    loss_components["net_heat_loss"] = net_heat_loss
    
    # Store components in inputs for step generator
    inputs["_loss_components"] = loss_components
    inputs["_delta_t"] = delta_t
    inputs["_volume"] = volume
    
    return net_heat_loss


def _generate_per_room_steps(inputs: Dict[str, Any], total_result: float, per_room_results: List[Dict[str, Any]]) -> List[str]:
    """Generate calculation steps for per-room Manual J calculations"""
    delta_t = inputs["indoor_design_temp"] - inputs["outdoor_design_temp"]
    
    steps = [
        f"Manual J Heat Loss Calculation (Per-Room)",
        f"Design Conditions: Indoor {inputs['indoor_design_temp']}°F, Outdoor {inputs['outdoor_design_temp']}°F",
        f"Temperature Difference (ΔT): {delta_t:.1f}°F",
        f"Ceiling Height: {inputs.get('ceiling_height', 8.0):.1f} ft",
        ""
    ]
    
    steps.append("Per-Room Heat Loss Requirements:")
    steps.append("-" * 70)
    
    for room in per_room_results:
        room_name = room['room_name']
        floor_area = room['floor_area']
        heat_loss = room['heat_loss_btu_hr']
        proportion = room['proportion_of_total']
        
        steps.append(f"{room_name:30s} {floor_area:8.1f} sq ft → {heat_loss:8.1f} BTU/hr ({proportion:5.1f}%)")
    
    steps.append("-" * 70)
    steps.append(f"{'TOTAL BUILDING':30s} {sum(r['floor_area'] for r in per_room_results):8.1f} sq ft → {total_result:8.1f} BTU/hr")
    
    return steps


def _generate_manual_j_steps(inputs: Dict[str, Any], result: float) -> List[str]:
    """Generate detailed calculation steps for Manual J heat loss"""
    
    # Check if this was a per-room calculation
    per_room_results = inputs.get("_per_room_results")
    if per_room_results:
        return _generate_per_room_steps(inputs, result, per_room_results)
    
    # Standard whole-building calculation steps
    components = inputs.get("_loss_components", {})
    delta_t = inputs.get("_delta_t", 0)
    volume = inputs.get("_volume", 0)
    
    # Determine if floor_area came from summing rooms
    floor_area_source = ""
    if inputs.get("rooms"):
        room_count = len(inputs["rooms"])
        floor_area_source = f" (summed from {room_count} rooms)"
    
    # Get actual floor_area used (after any summing)
    floor_area = inputs.get("floor_area")
    if floor_area is None or floor_area == 0:
        rooms = inputs.get("rooms") or []
        floor_area = sum(_room_area_sqft(r) for r in rooms) if isinstance(rooms, list) else 0.0
    
    steps = [
        f"Manual J Heat Loss Calculation (Whole-Building)",
        f"Design Conditions: Indoor {inputs['indoor_design_temp']}°F, Outdoor {inputs['outdoor_design_temp']}°F",
        f"Temperature Difference (ΔT): {delta_t:.1f}°F",
        f"Building Volume: {floor_area:.0f} sq ft{floor_area_source} × {inputs.get('ceiling_height', 8.0):.1f} ft = {volume:.0f} cu ft"
    ]
    
    # Conduction losses
    steps.append("\nConduction Losses (Q = U × A × ΔT):")
    
    if components.get("wall_loss", 0) > 0:
        wall_r = inputs.get("wall_r_value", 13.0)
        wall_area = inputs.get("wall_area", 0.0)
        steps.append(f"  Walls: U={1.0/wall_r:.4f} × {wall_area:.0f} sq ft × {delta_t:.1f}°F = {components['wall_loss']:.0f} BTU/hr")
    
    if components.get("roof_loss", 0) > 0:
        roof_r = inputs.get("roof_r_value", 30.0)
        roof_area_used = inputs.get("roof_area", floor_area)
        steps.append(f"  Roof/Ceiling: U={1.0/roof_r:.4f} × {roof_area_used:.0f} sq ft × {delta_t:.1f}°F = {components['roof_loss']:.0f} BTU/hr")
    
    if components.get("floor_loss", 0) > 0:
        floor_r = inputs.get("floor_r_value", 19.0)
        steps.append(f"  Floor: U={1.0/floor_r:.4f} × {floor_area:.0f} sq ft × {delta_t:.1f}°F = {components['floor_loss']:.0f} BTU/hr")
    
    if components.get("window_loss", 0) > 0:
        window_u = inputs.get("window_u_value", 0.5)
        window_area = inputs.get("window_area", 0.0)
        steps.append(f"  Windows: U={window_u:.2f} × {window_area:.0f} sq ft × {delta_t:.1f}°F = {components['window_loss']:.0f} BTU/hr")
    
    if components.get("door_loss", 0) > 0:
        door_u = inputs.get("door_u_value", 0.2)
        door_area = inputs.get("door_area", 0.0)
        steps.append(f"  Doors: U={door_u:.2f} × {door_area:.0f} sq ft × {delta_t:.1f}°F = {components['door_loss']:.0f} BTU/hr")
    
    total_conduction = components.get("total_conduction_loss", 0)
    steps.append(f"  Total Conduction Loss: {total_conduction:.0f} BTU/hr")
    
    # Infiltration
    if components.get("infiltration_loss", 0) > 0:
        ach = inputs.get("air_changes_per_hour", 0.5)
        steps.append(f"\nInfiltration Loss: 0.018 × {volume:.0f} cu ft × {ach:.2f} ACH × {delta_t:.1f}°F = {components['infiltration_loss']:.0f} BTU/hr")
    
    # Ventilation
    if components.get("ventilation_loss", 0) > 0:
        cfm = inputs.get("ventilation_cfm", 0.0)
        steps.append(f"Ventilation Loss: 1.08 × {cfm:.0f} CFM × {delta_t:.1f}°F = {components['ventilation_loss']:.0f} BTU/hr")
    
    # Total before gains
    total_before = components.get("total_loss_before_gains", 0)
    steps.append(f"\nTotal Heat Loss (before gains): {total_before:.0f} BTU/hr")
    
    # Internal gains
    if components.get("total_internal_gains", 0) > 0:
        steps.append("\nInternal Heat Gains (subtracted):")
        if components.get("occupant_gain", 0) > 0:
            steps.append(f"  Occupants: {inputs.get('occupant_count', 0)} × 400 BTU/hr = {components['occupant_gain']:.0f} BTU/hr")
        if components.get("appliance_gain", 0) > 0:
            steps.append(f"  Appliances: {components['appliance_gain']:.0f} BTU/hr")
        if components.get("lighting_gain", 0) > 0:
            steps.append(f"  Lighting: {components['lighting_gain']:.0f} BTU/hr")
        steps.append(f"  Total Internal Gains: {components['total_internal_gains']:.0f} BTU/hr")
    
    # Net result
    steps.append(f"\nNet Heat Loss: {total_before:.0f} - {components.get('total_internal_gains', 0):.0f} = {result:.0f} BTU/hr")
    
    return steps


def _room_area_sqft(room: Dict[str, Any]) -> float:
    """Return room area from room dict; accept 'floor_area', 'square_feet', or 'sq_ft'."""
    return float(room.get("floor_area") or room.get("square_feet") or room.get("sq_ft") or 0)


def _btu_hvac_total_sqft(inputs: Dict[str, Any]) -> float:
    """Total square footage for btu_hvac: single square_feet or sum of room areas (square_feet or sq_ft)."""
    single = inputs.get("square_feet")
    if single is not None and single > 0:
        return float(single)
    rooms = inputs.get("rooms") or []
    return sum(_room_area_sqft(r) for r in rooms) if isinstance(rooms, list) else 0.0


# Formula library structure
FORMULA_LIBRARY: Dict[str, Dict[str, Any]] = {
    # HVAC/BTU Calculations
    "btu_hvac": {
        "description": "Calculate BTU requirements for HVAC room sizing. Can handle 'square_feet' for a single area or a 'rooms' list for multiple rooms. Each room may use 'square_feet' or 'sq_ft' for area.",
        "formula": lambda inputs: (
            _btu_hvac_total_sqft(inputs)
            * inputs.get("btu_per_sqft", 25)
            * inputs.get("climate_factor", 1.0)
            * inputs.get("ceiling_height_factor", 1.0)
            * inputs.get("insulation_factor", 1.0)
        ),
        "required_inputs": [],  # validated in validation
        "optional_inputs": {
            "square_feet": None,
            "rooms": [],
            "btu_per_sqft": 25,  # Default for moderate climate
            "climate_factor": 1.0,  # 1.0 = moderate, 1.2-1.5 = hot, 0.8-0.9 = cold
            "ceiling_height_factor": 1.0,  # 1.0 = 8ft, adjust for taller ceilings
            "insulation_factor": 1.0,  # 1.0 = standard, <1.0 = well insulated, >1.0 = poor insulation
            "design_temp_f": None,  # Optional; document design temp (not used in simple formula)
        },
        "output_unit": "BTU/hr",
        "validation": lambda inputs: _btu_hvac_total_sqft(inputs) > 0,
        "step_generator": lambda inputs, result: (
            [
                f"Base calculation (rooms): Sum of room areas = {_btu_hvac_total_sqft(inputs):.0f} sq ft",
                f"Total area × {inputs.get('btu_per_sqft', 25)} BTU/sq ft = {_btu_hvac_total_sqft(inputs) * inputs.get('btu_per_sqft', 25):.0f} BTU/hr",
                f"Apply climate factor: × {inputs.get('climate_factor', 1.0)} = {_btu_hvac_total_sqft(inputs) * inputs.get('btu_per_sqft', 25) * inputs.get('climate_factor', 1.0):.0f} BTU/hr",
                f"Apply insulation factor: × {inputs.get('insulation_factor', 1.0)} = {result:.0f} BTU/hr"
            ] if not inputs.get("square_feet") else [
                f"Base calculation: {inputs['square_feet']} sq ft × {inputs.get('btu_per_sqft', 25)} BTU/sq ft = {inputs['square_feet'] * inputs.get('btu_per_sqft', 25):.0f} BTU/hr",
                f"Apply climate factor: × {inputs.get('climate_factor', 1.0)} = {inputs['square_feet'] * inputs.get('btu_per_sqft', 25) * inputs.get('climate_factor', 1.0):.0f} BTU/hr",
                f"Apply ceiling height factor: × {inputs.get('ceiling_height_factor', 1.0)} = {inputs['square_feet'] * inputs.get('btu_per_sqft', 25) * inputs.get('climate_factor', 1.0) * inputs.get('ceiling_height_factor', 1.0):.0f} BTU/hr",
                f"Apply insulation factor: × {inputs.get('insulation_factor', 1.0)} = {result:.0f} BTU/hr"
            ]
        )
    },
    "manual_j_heat_loss": {
        "description": "Calculate heat loss according to ACCA Manual J methodology. Can calculate whole-building heat loss OR per-room heat loss (set per_room_calculation=true). Includes conduction losses through building envelope, infiltration losses, ventilation losses, and subtracts internal heat gains.",
        "formula": lambda inputs: _calculate_manual_j_heat_loss(inputs),
        "required_inputs": ["outdoor_design_temp", "indoor_design_temp"],  # floor_area OR rooms required (validated in formula)
        "optional_inputs": {
            "floor_area": None,  # Total building area (OR provide rooms list)
            "rooms": [],  # List of rooms with floor_area or sq_ft (will be summed for whole-building OR calculated individually if per_room_calculation=true)
            "per_room_calculation": False,  # Set to true to get heat loss for EACH room individually (requires rooms list)
            "ceiling_height": 8.0,  # feet, default 8ft
            # Wall losses
            "wall_area": 0.0,  # sq ft
            "wall_r_value": 13.0,  # R-value, default R-13
            # Roof/Ceiling losses
            "roof_area": 0.0,  # sq ft (if 0, uses floor_area)
            "roof_r_value": 30.0,  # R-value, default R-30
            # Floor losses
            "floor_r_value": 19.0,  # R-value, default R-19
            "floor_over_unconditioned": False,  # True if floor over unconditioned space
            # Window losses
            "window_area": 0.0,  # sq ft
            "window_u_value": 0.5,  # U-value, default U-0.5 (double pane)
            # Door losses
            "door_area": 0.0,  # sq ft
            "door_u_value": 0.2,  # U-value, default U-0.2 (insulated door)
            # Infiltration
            "air_changes_per_hour": 0.5,  # ACH, default 0.5 (tight construction)
            # Ventilation
            "ventilation_cfm": 0.0,  # CFM of mechanical ventilation
            # Internal heat gains (subtracted from losses)
            "occupant_count": 0,  # number of occupants
            "appliance_heat_gain": 0.0,  # BTU/hr from appliances
            "lighting_heat_gain": 0.0,  # BTU/hr from lighting
        },
        "output_unit": "BTU/hr",
        "validation": lambda inputs: (
            inputs["outdoor_design_temp"] < inputs["indoor_design_temp"] and
            (inputs.get("ceiling_height") or 8.0) > 0 and
            # Either floor_area or rooms must be provided (floor_area can be None after optional defaults)
            ((inputs.get("floor_area") or 0) > 0 or (isinstance(inputs.get("rooms"), list) and len(inputs.get("rooms", [])) > 0))
        ),
        "step_generator": lambda inputs, result: _generate_manual_j_steps(inputs, result)
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


def get_formula_catalog_for_agents() -> List[Dict[str, Any]]:
    """
    Return an agent-facing catalog of formulas with EXACT specifications from FORMULA_LIBRARY.
    
    This exposes the actual parameter names, types, defaults, and format requirements
    so agents can construct inputs correctly without hallucinating parameters.
    """
    return [
        {
            "name": "manual_j_heat_loss",
            "description": "ACCA Manual J heat loss calculation. Can calculate WHOLE-BUILDING heat loss OR PER-ROOM heat loss (set per_room_calculation=true). Uses indoor/outdoor design temp, floor area, and envelope details. Use when the user wants to verify or compute heat loss from design temperature.",
            "when_to_use": [
                "verify or check BTU/heat loss numbers against design temp",
                "design temp and square footage",
                "Manual J",
                "actual heat loss",
                "heat loss for this building or per-room",
                "calculate heat loss",
                "what SHOULD the BTU be (with design temp)",
                "per-room BTU requirements",
                "radiator sizing",
            ],
            "required_inputs": {
                "outdoor_design_temp": {"type": "number", "description": "Outdoor design temperature (°F), e.g., 4.1"},
                "indoor_design_temp": {"type": "number", "description": "Indoor design temperature (°F), typically 70"},
            },
            "optional_inputs": {
                "floor_area": {"type": "number", "default": None, "description": "Total building floor area (sq ft). OR provide 'rooms' list."},
                "rooms": {"type": "array", "default": [], "description": "List of rooms. Each room: {name: string, floor_area: number} OR {name: string, sq_ft: number}. Will be summed for whole-building OR calculated individually if per_room_calculation=true."},
                "per_room_calculation": {"type": "boolean", "default": False, "description": "Set to TRUE to get heat loss for EACH room individually (requires rooms list). Returns per-room BTU requirements for radiator sizing."},
                "ceiling_height": {"type": "number", "default": 8.0, "description": "Average ceiling height (feet)"},
                "wall_area": {"type": "number", "default": 0.0, "description": "Total exterior wall area (sq ft)"},
                "wall_r_value": {"type": "number", "default": 13.0, "description": "Wall insulation R-value"},
                "roof_area": {"type": "number", "default": 0.0, "description": "Roof area (sq ft). If 0, uses floor_area."},
                "roof_r_value": {"type": "number", "default": 30.0, "description": "Roof/ceiling insulation R-value"},
                "floor_r_value": {"type": "number", "default": 19.0, "description": "Floor insulation R-value"},
                "floor_over_unconditioned": {"type": "boolean", "default": False, "description": "True if floor is over unconditioned space"},
                "window_area": {"type": "number", "default": 0.0, "description": "Total window area (sq ft)"},
                "window_u_value": {"type": "number", "default": 0.5, "description": "Window U-value (default 0.5 for double-pane)"},
                "door_area": {"type": "number", "default": 0.0, "description": "Total door area (sq ft)"},
                "door_u_value": {"type": "number", "default": 0.2, "description": "Door U-value (default 0.2 for insulated door)"},
                "air_changes_per_hour": {"type": "number", "default": 0.5, "description": "Air changes per hour (ACH), default 0.5 for tight construction"},
                "ventilation_cfm": {"type": "number", "default": 0.0, "description": "Mechanical ventilation rate (CFM)"},
                "occupant_count": {"type": "number", "default": 0, "description": "Number of occupants (for internal heat gain)"},
                "appliance_heat_gain": {"type": "number", "default": 0.0, "description": "Heat gain from appliances (BTU/hr)"},
                "lighting_heat_gain": {"type": "number", "default": 0.0, "description": "Heat gain from lighting (BTU/hr)"},
            },
            "output_unit": "BTU/hr",
            "important_notes": [
                "For rooms list, use ONLY these keys per room: 'name', 'floor_area' (or 'sq_ft'). Do NOT add other fields.",
                "outdoor_design_temp is the design temp from the document (e.g., 4.1°F)",
                "indoor_design_temp is typically 70°F unless document specifies otherwise",
                "ONLY use parameter NAMES listed in required_inputs and optional_inputs above",
                "Do NOT invent parameter NAMES like 'insulation' or 'insulation_quality' - use wall_r_value, roof_r_value instead",
                "**INTERPRET qualitative descriptions**: 'low insulation'→wall_r_value:10, '1970s window'→window_u_value:0.7, 'drafty'→air_changes_per_hour:1.0",
                "**Make intelligent estimates** from document context rather than refusing to calculate",
                "**For per-room BTU requirements (e.g., radiator sizing): set per_room_calculation=true with rooms list**",
                "**For whole-building heat loss: leave per_room_calculation=false (default) or omit it**",
                "Each room is treated as its own building when per_room_calculation=true",
            ],
        },
        {
            "name": "btu_hvac",
            "description": "Rule-of-thumb BTU sizing: total area × BTU per sq ft × factors. Does NOT use design temperature. Use for rough total BTU need or room sizing when design-temp verification is not requested.",
            "when_to_use": [
                "rough BTU requirements",
                "total BTU for these rooms",
                "HVAC sizing (no design temp)",
            ],
            "required_inputs": {},  # Either square_feet or rooms is required, validated in formula
            "optional_inputs": {
                "square_feet": {"type": "number", "default": None, "description": "Total area (sq ft). OR provide 'rooms' list."},
                "rooms": {"type": "array", "default": [], "description": "List of rooms. Each room: {name: string, square_feet: number} OR {name: string, sq_ft: number} OR {name: string, floor_area: number}. Will be summed."},
                "btu_per_sqft": {"type": "number", "default": 25, "description": "BTU per square foot (default 25 for moderate climate)"},
                "climate_factor": {"type": "number", "default": 1.0, "description": "Climate adjustment (1.0=moderate, 1.2-1.5=hot, 0.8-0.9=cold)"},
                "ceiling_height_factor": {"type": "number", "default": 1.0, "description": "Ceiling height factor (1.0 for 8ft, adjust for taller)"},
                "insulation_factor": {"type": "number", "default": 1.0, "description": "Insulation factor (1.0=standard, <1.0=well insulated, >1.0=poor)"},
                "design_temp_f": {"type": "number", "default": None, "description": "Optional design temp (not used in calculation, just for reference)"},
            },
            "output_unit": "BTU/hr",
            "important_notes": [
                "This is a simple rule-of-thumb calculation, NOT Manual J",
                "Does NOT use design temperature in calculations",
                "For design-temp-based verification, use manual_j_heat_loss instead",
            ],
        },
        {
            "name": "ohms_law_voltage",
            "description": "Voltage = current × resistance (V = I × R).",
            "when_to_use": ["voltage from current and resistance"],
            "required_inputs": {
                "current": {"type": "number", "description": "Current in amperes (A)"},
                "resistance": {"type": "number", "description": "Resistance in ohms (Ω)"},
            },
            "optional_inputs": {},
            "output_unit": "volts",
        },
        {
            "name": "ohms_law_current",
            "description": "Current = voltage / resistance (I = V / R).",
            "when_to_use": ["current from voltage and resistance"],
            "required_inputs": {
                "voltage": {"type": "number", "description": "Voltage in volts (V)"},
                "resistance": {"type": "number", "description": "Resistance in ohms (Ω)"},
            },
            "optional_inputs": {},
            "output_unit": "amps",
        },
        {
            "name": "ohms_law_resistance",
            "description": "Resistance = voltage / current (R = V / I).",
            "when_to_use": ["resistance from voltage and current"],
            "required_inputs": {
                "voltage": {"type": "number", "description": "Voltage in volts (V)"},
                "current": {"type": "number", "description": "Current in amperes (A)"},
            },
            "optional_inputs": {},
            "output_unit": "ohms",
        },
    ]


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
        
        # Validate that no unknown parameters are passed
        optional_inputs = formula_def.get("optional_inputs", {})
        allowed_params = set(required_inputs) | set(optional_inputs.keys())
        # Filter out internal fields that start with underscore
        provided_params = {k for k in inputs.keys() if not k.startswith('_')}
        unknown_params = provided_params - allowed_params
        if unknown_params:
            logger.warning(f"⚠️ Unknown parameters passed to {formula_name}: {unknown_params}")
            return {
                "result": None,
                "unit": None,
                "formula_used": formula_name,
                "steps": [],
                "success": False,
                "error": f"Unknown parameters: {', '.join(unknown_params)}. Valid parameters are: {', '.join(sorted(allowed_params))}"
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

