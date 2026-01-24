"""
Parameter validation and type conversion for preprocessors
"""

from typing import Dict, Any, List, Optional
from loguru import logger


def validate_and_convert_params(
    params: Dict[str, Any], parameter_definitions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate and convert preprocessor parameters to their correct types.

    Converts string representations of numbers to int/float as needed,
    validates against min/max bounds, and ensures boolean values are correct.

    Args:
        params: Dictionary of parameter names to values (may include strings)
        parameter_definitions: List of parameter definition dictionaries from the registry

    Returns:
        Dictionary with validated and converted parameters

    Raises:
        ValueError: If a parameter fails validation
    """
    validated_params = {}

    # Create a lookup dict for parameter definitions
    param_defs = {p["name"]: p for p in parameter_definitions}

    for param_name, param_value in params.items():
        # Skip if parameter not in definitions (allow extra params to pass through)
        if param_name not in param_defs:
            logger.warning(
                f"Parameter '{param_name}' not found in definitions, passing through as-is"
            )
            validated_params[param_name] = param_value
            continue

        param_def = param_defs[param_name]
        param_type = param_def.get("type")

        try:
            # Handle None values - use default if available
            if param_value is None:
                if "default" in param_def:
                    validated_params[param_name] = param_def["default"]
                continue

            # Convert and validate based on type
            if param_type == "int":
                validated_value = _convert_to_int(param_value, param_name)
                validated_value = _validate_numeric_bounds(
                    validated_value, param_def, param_name
                )
                validated_params[param_name] = validated_value

            elif param_type == "float":
                validated_value = _convert_to_float(param_value, param_name)
                validated_value = _validate_numeric_bounds(
                    validated_value, param_def, param_name
                )
                validated_params[param_name] = validated_value

            elif param_type == "bool":
                validated_params[param_name] = _convert_to_bool(param_value, param_name)

            elif param_type == "category":
                validated_params[param_name] = _validate_category(
                    param_value, param_def, param_name
                )

            else:
                # Unknown type, pass through as-is
                validated_params[param_name] = param_value

        except ValueError as e:
            raise ValueError(f"Validation error for parameter '{param_name}': {str(e)}")

    return validated_params


def _convert_to_int(value: Any, param_name: str) -> int:
    """Convert a value to int, handling string representations."""
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        try:
            # Handle string representations like "512", "1024"
            return int(value)
        except ValueError:
            try:
                # Try converting via float first (handles "512.0")
                return int(float(value))
            except ValueError:
                raise ValueError(f"Cannot convert '{value}' to integer")

    if isinstance(value, float):
        return int(value)

    raise ValueError(f"Cannot convert type {type(value).__name__} to integer")


def _convert_to_float(value: Any, param_name: str) -> float:
    """Convert a value to float, handling string representations."""
    if isinstance(value, float):
        return value

    if isinstance(value, (int, str)):
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot convert '{value}' to float")

    raise ValueError(f"Cannot convert type {type(value).__name__} to float")


def _convert_to_bool(value: Any, param_name: str) -> bool:
    """Convert a value to bool, handling various representations."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lower_val = value.lower().strip()
        if lower_val in ("true", "1", "yes", "on"):
            return True
        elif lower_val in ("false", "0", "no", "off"):
            return False
        else:
            raise ValueError(f"Cannot convert string '{value}' to boolean")

    if isinstance(value, (int, float)):
        return bool(value)

    raise ValueError(f"Cannot convert type {type(value).__name__} to boolean")


def _validate_numeric_bounds(
    value: float, param_def: Dict[str, Any], param_name: str
) -> float:
    """Validate that a numeric value is within specified bounds."""
    min_val = param_def.get("min")
    max_val = param_def.get("max")

    if min_val is not None and value < min_val:
        raise ValueError(f"Value {value} is below minimum {min_val}")

    if max_val is not None and value > max_val:
        raise ValueError(f"Value {value} is above maximum {max_val}")

    return value


def _validate_category(value: Any, param_def: Dict[str, Any], param_name: str) -> Any:
    """Validate that a value is in the allowed category options."""
    options = param_def.get("options", [])

    if not options:
        # No options defined, accept any value
        return value

    # Extract valid values from options
    valid_values = [opt.get("value") for opt in options]

    # Handle string representations of numbers for category types
    if isinstance(value, str):
        # Try to match as-is first
        if value in valid_values:
            return value

        # Try numeric conversion if valid values are numeric
        for valid_value in valid_values:
            if isinstance(valid_value, (int, float)):
                try:
                    converted = (
                        float(value) if isinstance(valid_value, float) else int(value)
                    )
                    if converted in valid_values:
                        return converted
                except ValueError:
                    pass

    if value not in valid_values:
        raise ValueError(f"Value '{value}' not in allowed options: {valid_values}")

    return value
