"""JSON values that keep Python ints, floats, and bools distinct."""

from __future__ import annotations

import math
from typing import Any


def enc(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return {"bool": value}
    if isinstance(value, int):
        return {"int": str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "inf" if value > 0 else "-inf"}
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return {"float": "-0.0"}
        return {"float": repr(value)}
    if isinstance(value, str):
        return {"str": value}
    if isinstance(value, list):
        return [enc(v) for v in value]
    if isinstance(value, tuple):
        return {"tuple": [enc(v) for v in value]}
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: enc(item) for key, item in value.items()}
        return {"map": [[enc(key), enc(item)] for key, item in value.items()]}
    raise TypeError(f"cannot encode {type(value).__name__}")
