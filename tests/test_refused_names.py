"""A refused name raises an error that says its one spelling, accessed or imported; every spelling it names exists."""
import importlib
import re

import pytest

from dmipy_design._moved import SPELLINGS

REFUSED = [(module, name, spelling) for module, names in SPELLINGS.items() for name, spelling in names.items()]
IDS = [f"{module}.{name}" for module, name, _ in REFUSED]


@pytest.mark.parametrize("module,name,spelling", REFUSED, ids=IDS)
def test_accessing_a_refused_name_says_its_spelling(module, name, spelling):
    m = importlib.import_module(module)
    with pytest.raises(AttributeError, match=re.escape(f"{module}.{name} does not exist: use {spelling}")):
        getattr(m, name)
    assert not hasattr(m, name) and getattr(m, name, None) is None and name not in getattr(m, "__all__", ())


@pytest.mark.parametrize("module,name,spelling", REFUSED, ids=IDS)
def test_importing_a_refused_name_says_its_spelling(module, name, spelling):
    with pytest.raises(ImportError, match=re.escape(f"{module}.{name} does not exist: use {spelling}")):
        exec(f"from {module} import {name}", {})


@pytest.mark.parametrize("module,name,spelling", REFUSED, ids=IDS)
def test_the_spelling_a_refusal_names_exists(module, name, spelling):
    new, where = re.fullmatch(r"(\w+) \(from ([\w.]+) import \1\)", spelling).groups()
    assert hasattr(importlib.import_module(where), new) and new in importlib.import_module(where).__all__


def test_a_name_that_was_never_defined_is_the_ordinary_attribute_error():
    certify = importlib.import_module("dmipy_design.certify")
    with pytest.raises(AttributeError, match="has no attribute 'no_such_name'"):
        certify.no_such_name
    with pytest.raises(ImportError, match="cannot import name 'no_such_name'"):
        exec("from dmipy_design.certify import no_such_name", {})
