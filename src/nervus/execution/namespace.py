"""An on-demand metadata view of a live namespace; never an object-content index."""

import json
from types import FunctionType


_TYPE_NAME = type.__dict__["__name__"]
_LENGTH_TYPES = (str, bytes, bytearray, list, tuple, dict, set, frozenset, range)


def validate_limits(max_entries, max_bytes, prefix):
    if type(max_entries) is not int or not 0 <= max_entries <= 100:
        raise ValueError("max_entries must be between 0 and 100")
    if type(max_bytes) is not int or not 256 <= max_bytes <= 16384:
        raise ValueError("max_bytes must be between 256 and 16384")
    if type(prefix) is not str or len(prefix) > 128:
        raise ValueError("prefix must be text of at most 128 characters")


def _short(value, limit):
    # Native type metadata can be a str subclass; never format that subclass.
    if type(value) is not str:
        return "<metadata unavailable>", True
    clipped = len(value) > limit
    value = value[:limit].encode("utf-8", errors="backslashreplace").decode("utf-8")
    clipped |= len(value) > limit
    return (value[:limit] + "…", True) if clipped else (value, False)


def _function_signature(function):
    # Only exact Python functions reach here. Do not inspect __signature__,
    # __wrapped__, annotations (which can be lazy), default values or closures.
    code = function.__code__
    names = code.co_varnames
    defaults = function.__defaults__
    default_count = tuple.__len__(defaults) if defaults is not None else 0
    kwdefaults = function.__kwdefaults__
    parts = []
    partial = code.co_argcount + code.co_kwonlyargcount > 16
    for index in range(min(code.co_argcount, 16)):
        name, clipped = _short(names[index], 64)
        partial |= clipped
        parts.append(name + ("=…" if index >= code.co_argcount - default_count else ""))
        if index + 1 == code.co_posonlyargcount:
            parts.append("/")
    offset = code.co_argcount + code.co_kwonlyargcount
    if code.co_flags & 0x04:  # CO_VARARGS
        name, clipped = _short(names[offset], 64)
        parts.append("*" + name)
        partial |= clipped
        offset += 1
    elif code.co_kwonlyargcount:
        parts.append("*")
    for index in range(min(code.co_kwonlyargcount, max(0, 16 - code.co_argcount))):
        name, clipped = _short(names[code.co_argcount + index], 64)
        partial |= clipped
        # Exact-string keys only: dict lookup against hostile keys could call __eq__.
        has_default = (type(kwdefaults) is dict
                       and any(type(key) is str and key == name for key in dict.keys(kwdefaults)))
        if kwdefaults is not None and type(kwdefaults) is not dict:
            partial = True
        parts.append(name + ("=…" if has_default else ""))
    if code.co_flags & 0x08:  # CO_VARKEYWORDS
        name, clipped = _short(names[offset], 64)
        parts.append("**" + name)
        partial |= clipped
    if partial:
        parts.append("…")
    signature, clipped = _short("(" + ", ".join(parts) + ")", 256)
    return signature, partial or clipped


def namespace_directory(namespace, *, hidden, reference_info, max_entries=50, max_bytes=8192, prefix=""):
    validate_limits(max_entries, max_bytes, prefix)
    result = {"entries": [], "truncated": False}
    for name, value in namespace.items():
        if type(name) is not str or not name.startswith(prefix):
            continue
        if name in hidden and value is hidden[name]:
            continue
        if len(result["entries"]) >= max_entries:
            result["truncated"] = True
            break
        cls = type(value)  # Does not ask value.__class__ or a custom metaclass.
        label, name_clipped = _short(name, 96)
        type_name, type_clipped = _short(_TYPE_NAME.__get__(cls), 96)
        entry = {"name": label, "type": type_name, "kind": "value"}
        partial = name_clipped or type_clipped
        reference = reference_info(value)
        if reference is not None:
            entry.update(kind="capability_reference", **reference)
        elif cls is FunctionType:
            entry["kind"] = "function"
            entry["signature"], clipped = _function_signature(value)
            partial |= clipped
        elif any(cls is allowed for allowed in _LENGTH_TYPES):
            try:
                entry["length"] = len(value)  # Exact builtins, never a subclass hook.
            except OverflowError:
                entry["length_unavailable"] = True
        if partial:
            entry["metadata_truncated"] = True
        # Reserve the longer false spelling, so the complete return value fits.
        trial = {"entries": [*result["entries"], entry], "truncated": False}
        if len(json.dumps(trial, ensure_ascii=True, separators=(",", ":")).encode("ascii")) > max_bytes:
            result["truncated"] = True
            break
        result["entries"].append(entry)
        result["truncated"] |= partial
    return result
