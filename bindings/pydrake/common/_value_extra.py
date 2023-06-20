# See `ExecuteExtraPythonCode` in `pydrake_pybind.h` for usage details and
# rationale.

from pydrake.common.cpp_param import List


def _AbstractValue_Make(value):
    """Returns an AbstractValue containing the given ``value``.
    The factory method ``AbstractValue.Make(value)`` and the constructor
    ``Value(value)`` are equivalent."""
    if isinstance(value, list) and len(value) > 0:
        inner_cls = type(value[0])
        cls = List[inner_cls]
    else:
        cls = type(value)
    value_cls, _ = Value.get_instantiation(cls, throw_error=False)
    if value_cls is None:
        value_cls = Value[object]
    return value_cls(value)


AbstractValue.Make = _AbstractValue_Make

# Adjust how type inference works for Value(some_value). We want to use the
# precise semantics of AbstractValue.Make, not the anything-goes TemplateBase.
setattr(Value, "_call_internal", _AbstractValue_Make)
