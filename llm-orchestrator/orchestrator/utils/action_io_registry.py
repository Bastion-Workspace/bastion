"""
Action I/O Registry - Typed contracts for tools used by Agent Factory and Workflow Composer.

Every tool registers with register_action(); the registry provides schema generation,
type compatibility checks, and field introspection for UI and pipeline execution.
"""

from pydantic import BaseModel
from typing import Any, Callable, Dict, List, Optional, Type, get_origin, get_args

# Optional json import for schema; avoid top-level if not needed
import json  # noqa: F401 - used by model_json_schema


class ActionContract(BaseModel):
    """Complete I/O contract for a registered action."""

    name: str
    category: str
    description: str
    short_description: Optional[str] = None
    inputs_model: Type[BaseModel]
    params_model: Optional[Type[BaseModel]] = None
    outputs_model: Type[BaseModel]
    tool_function: Callable[..., Any]
    retriable: bool = True

    class Config:
        arbitrary_types_allowed = True

    def get_input_schema(self) -> dict:
        """JSON Schema for inputs (Workflow Composer wiring UI)."""
        return self.inputs_model.model_json_schema()

    def get_params_schema(self) -> dict:
        """JSON Schema for params (Workflow Composer config panel)."""
        if self.params_model:
            return self.params_model.model_json_schema()
        return {}

    def get_output_schema(self) -> dict:
        """JSON Schema for outputs (downstream wiring)."""
        return self.outputs_model.model_json_schema()

    def get_output_fields(self) -> List[Dict[str, Any]]:
        """
        Output fields as list of {name, type, description} for Workflow Composer
        dropdown when wiring downstream step inputs.
        """
        fields = []
        for name, field_info in self.outputs_model.model_fields.items():
            if name == "formatted":
                continue
            field_type = _python_type_to_tool_type(field_info.annotation)
            fields.append({
                "name": name,
                "type": field_type,
                "description": field_info.description or "",
            })
        return fields

    def get_input_fields(self) -> List[Dict[str, Any]]:
        """Input fields for the Workflow Composer wiring UI."""
        fields = []
        for name, field_info in self.inputs_model.model_fields.items():
            field_type = _python_type_to_tool_type(field_info.annotation)
            required = field_info.is_required()
            default = None
            if not required and field_info.default is not None:
                default = field_info.default
            fields.append({
                "name": name,
                "type": field_type,
                "description": field_info.description or "",
                "required": required,
                "default": default,
            })
        return fields


def _python_type_to_tool_type(python_type: Any) -> str:
    """Map Python type annotations to the tool type system."""
    origin = get_origin(python_type)
    args = get_args(python_type) if python_type is not None else ()

    if python_type is str:
        return "text"
    if python_type in (int, float):
        return "number"
    if python_type is bool:
        return "boolean"
    if origin is list:
        inner = args[0] if args else Any
        inner_str = _python_type_to_tool_type(inner)
        return f"list[{inner_str}]"
    if origin is dict:
        return "record"
    if python_type is type(None):
        return "any"
    try:
        from datetime import datetime, date
        if python_type in (datetime, date):
            return "date"
        if origin is type(None) or (get_origin(python_type) is not None):
            for a in args:
                if a is not type(None) and a in (datetime, date):
                    return "date"
    except (TypeError, ImportError):
        pass
    try:
        if isinstance(python_type, type) and issubclass(python_type, BaseModel):
            if getattr(python_type, "__name__", "") == "FileRef":
                return "file_ref"
            return "record"
    except TypeError:
        pass
    type_name = getattr(python_type, "__name__", "")
    if type_name in ("datetime", "date"):
        return "date"
    if origin is not None and args:
        for a in args:
            if getattr(a, "__name__", "") in ("datetime", "date"):
                return "date"
    return "any"


# ── Global Registry ──────────────────────────────────────

_REGISTRY: Dict[str, ActionContract] = {}


def register_action(
    name: str,
    category: str,
    description: str,
    inputs_model: Type[BaseModel],
    outputs_model: Type[BaseModel],
    tool_function: Callable[..., Any],
    params_model: Optional[Type[BaseModel]] = None,
    short_description: Optional[str] = None,
    retriable: bool = True,
) -> None:
    """Register a tool with its typed I/O contract.
    description: Full text for LLM tool binding (may include instructions).
    short_description: Optional one-liner for UI pickers; if set, shown instead of description in the composer.
    """
    _REGISTRY[name] = ActionContract(
        name=name,
        category=category,
        description=description,
        short_description=short_description,
        inputs_model=inputs_model,
        params_model=params_model,
        outputs_model=outputs_model,
        tool_function=tool_function,
        retriable=retriable,
    )


def get_action(name: str) -> Optional[ActionContract]:
    """Look up a registered action by name."""
    return _REGISTRY.get(name)


def get_all_actions() -> Dict[str, ActionContract]:
    """Return the full registry."""
    return _REGISTRY.copy()


def get_actions_by_category(category: str) -> Dict[str, ActionContract]:
    """Return all actions in a category."""
    return {k: v for k, v in _REGISTRY.items() if v.category == category}


def get_categories_for_tools(tool_names: List[str]) -> List[str]:
    """Map tool function names to unique categories via the registry."""
    categories: set = set()
    for name in tool_names:
        action_name = name[:-5] if name.endswith("_tool") else name
        contract = _REGISTRY.get(action_name)
        if contract:
            categories.add(contract.category)
    return sorted(categories)


def get_compatible_upstream_fields(
    target_input_type: str,
    upstream_outputs: Dict[str, str],
) -> List[str]:
    """
    Given a target input type and a dict of {field_name: field_type} from
    upstream steps, return which fields are compatible (with coercion).
    """
    compatible = []
    for field_name, field_type in upstream_outputs.items():
        if is_type_compatible(source_type=field_type, target_type=target_input_type):
            compatible.append(field_name)
    return compatible


def is_type_compatible(source_type: str, target_type: str) -> bool:
    """
    Check if source_type can be wired to target_type (with coercion).
    Returns True if the connection is valid.
    """
    if source_type == target_type:
        return True

    if target_type == "any" or source_type == "any":
        return True

    if target_type == "text":
        return True

    if source_type == "text":
        if target_type.startswith("list["):
            return True
        return target_type in ("number", "boolean", "date", "record", "any")

    if source_type == "number" and target_type == "boolean":
        return True

    if source_type == "file_ref":
        return target_type in ("text", "record", "any")

    if source_type.startswith("list[") and target_type == "text":
        return True
    if source_type.startswith("list[") and target_type.startswith("list["):
        inner_source = source_type[5:-1]
        inner_target = target_type[5:-1]
        return is_type_compatible(inner_source, inner_target)

    if source_type == "record" and target_type == "text":
        return True

    if source_type == "text" and target_type.startswith("list["):
        return True

    return False
