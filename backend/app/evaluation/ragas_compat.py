"""Import-time compatibility layer for RAGAS + langchain-community 0.4.

``ragas`` (every release tested: 0.2.15, 0.3.9, 0.4.3) imports
``langchain_community.chat_models.vertexai`` at module level. With
``langchain-community`` >= 0.4 that module no longer exists (the VertexAI
wrappers were extracted to standalone Google packages), so importing ragas
fails before any metric code runs.

The import inside ragas is only used to list ``ChatVertexAI`` in its
``MULTIPLE_COMPLETION_SUPPORTED`` registry — nothing our four metrics ever
instantiates. This module therefore registers a minimal alias module into
``sys.modules`` **before ragas is imported** so the import succeeds. The stub
``ChatVertexAI`` raises loudly if it is ever instantiated: it is never a
silent replacement for a real model.

Import this module before ``import ragas`` anywhere in the evaluation package.
"""

from __future__ import annotations

import importlib
import sys
import types

_STUB_MODULE_NAME = "langchain_community.chat_models.vertexai"


def install_vertexai_stub() -> None:
    """Register the alias module ragas imports, unless the real one exists.

    If a future ``langchain-community`` (or another installed package) provides
    the real module again, it is left untouched.
    """
    if _STUB_MODULE_NAME in sys.modules:
        return
    try:
        importlib.import_module(_STUB_MODULE_NAME)
        return  # the real module is available - nothing to do
    except ImportError:
        pass

    stub = types.ModuleType(_STUB_MODULE_NAME)
    stub.__doc__ = (
        "Compatibility shim for ragas (which imports this module name at "
        "module level). The real ChatVertexAI moved to "
        "langchain-google-vertexai; evaluation never uses it."
    )

    class ChatVertexAI:  # pragma: no cover - explicit loud failure
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "ChatVertexAI is not installed and is not used by the "
                "evaluation metrics. Install langchain-google-vertexai if you "
                "actually need it."
            )

    stub.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules[_STUB_MODULE_NAME] = stub


def import_ragas():
    """Import ragas with the compatibility shim in place.

    Returns the ragas module; raises the original ImportError if ragas itself
    is missing (a real, explicit failure - never silently swallowed).
    """
    install_vertexai_stub()
    importlib.import_module("ragas")
    return sys.modules["ragas"]


# Importing this module performs the registration immediately, so a plain
# ``import ragas`` in the same process afterwards also works.
install_vertexai_stub()
