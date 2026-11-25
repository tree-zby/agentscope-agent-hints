"""Agent Hint protocol v1 for AgentScope 2.x.

``agentscope_agent_hints`` ships the protocol core (pure functions, no
AgentScope import) plus one adapter module:

* :mod:`agentscope_agent_hints.adapter` — ``HintedChatModel`` and
  ``AgentHintLifecycle``.

Importing this package never imports AgentScope, so the core can be tested
and reused on its own.
"""

from .encoder import HintDecodeError, decode, encode
from .manage import ManageClient, ManageResult
from .protocol import (
    HEADER_NAME,
    HintValidationError,
    is_manage_hint,
    validate,
    validate_or_raise,
)
from .state import (
    HintConfig,
    HintSession,
    MessageView,
    RequestView,
    SessionIsolatedError,
    SessionSnapshot,
    new_session_id,
)

__version__ = "0.1.0"

__all__ = [
    "HEADER_NAME",
    "HintConfig",
    "HintDecodeError",
    "HintSession",
    "HintValidationError",
    "ManageClient",
    "ManageResult",
    "MessageView",
    "RequestView",
    "SessionIsolatedError",
    "SessionSnapshot",
    "__version__",
    "decode",
    "encode",
    "is_manage_hint",
    "new_session_id",
    "validate",
    "validate_or_raise",
]
