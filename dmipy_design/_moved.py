"""The names dmipy-design refuses by name, each with its one spelling.

A module that refuses names defines ``__getattr__`` through :func:`refuse`: ``dmipy_design.certify.replay_envelope``
raises an ``AttributeError`` that says ``deliverable_set``, and ``from dmipy_design.certify import replay_envelope``
raises an ``ImportError`` that says it too (CPython drops the ``AttributeError`` of a module's ``__getattr__`` inside a
``from`` import and reports only that the name cannot be imported). A name that is none of these is the ordinary
``AttributeError``."""
import dis
import sys

SPELLINGS = {
    "dmipy_design.certify": {
        "ReplayEnvelope": "DeliverableSet (from dmipy_design.certify import DeliverableSet)",
        "replay_envelope": "deliverable_set (from dmipy_design.certify import deliverable_set)",
    },
}
"""``{module: {refused name: its one spelling}}``, stated whole: a new name with the import that brings it."""


def _from_import():
    """Whether the lookup that ``refuse`` answers is the one of a ``from module import name`` statement: the
    statement's frame is three frames up (``refuse`` is called by a module's ``__getattr__``, which the statement
    runs), standing at its ``IMPORT_FROM`` instruction."""
    try:
        frame = sys._getframe(3)
    except ValueError:
        return False
    return frame.f_code.co_code[frame.f_lasti] == dis.opmap["IMPORT_FROM"]


def refuse(module, name):
    """The exception of ``module.name``: the refusal that says the spelling of a refused name (an ``ImportError`` in a
    ``from module import name`` statement, an ``AttributeError`` otherwise, so ``hasattr`` stays false), the ordinary
    ``AttributeError`` of any other name. Called as ``raise refuse(__name__, name)`` by a module's ``__getattr__``."""
    spelling = SPELLINGS.get(module, {}).get(name)
    if spelling is None:
        return AttributeError(f"module {module!r} has no attribute {name!r}")
    message = f"{module}.{name} does not exist: use {spelling}"
    return ImportError(message, name=module) if _from_import() else AttributeError(message)
