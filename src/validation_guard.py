"""Hard guard on the validation split.

agent.md §12 forbids tuning on eval, and D12 reserves exactly one look at the
validation ceilings. Comments and discipline are not enough: any code path that
reads the validation split now has to be explicitly unlocked, and every attempt --
allowed or refused -- is appended to results/validation_access.log.

Unlock with `--allow-validation` on main.py, or ALLOW_VALIDATION=1 in the
environment for a standalone script. Both require a reason.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "validation_access.log"

_allowed = False
_reason = ""


class ValidationAccessError(PermissionError):
    """Raised when something tries to read validation without being unlocked."""


def _append(verdict: str, context: str, reason: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{verdict}\t{context}\t{reason}\n")


def allow(reason: str) -> None:
    """Unlock validation access for this process. Logged."""
    global _allowed, _reason
    _allowed, _reason = True, reason
    _append("UNLOCK", "process", reason)


def is_allowed() -> bool:
    if _allowed:
        return True
    return os.environ.get("ALLOW_VALIDATION", "") in ("1", "true", "yes")


def _log_label() -> str:
    """LOG is not guaranteed to live under ROOT (tests redirect it), so never
    assume relative_to() will succeed -- it raises ValueError when it does not."""
    try:
        return str(LOG.relative_to(ROOT))
    except ValueError:
        return str(LOG)


def require(context: str) -> None:
    """Call before reading anything from the validation split.

    Raises ValidationAccessError unless unlocked. Every call is logged either way,
    so the log is the audit trail: if it is empty, validation was never read.
    """
    reason = _reason or os.environ.get("ALLOW_VALIDATION_REASON", "")
    if not is_allowed():
        _append("DENIED", context, reason or "(no reason given)")
        raise ValidationAccessError(
            f"Refusing to read the validation split from {context!r}.\n"
            "agent.md §12 forbids tuning on eval and D12 reserves one measured look.\n"
            "If this read is genuinely intended, pass --allow-validation to main.py "
            "(or set ALLOW_VALIDATION=1 plus ALLOW_VALIDATION_REASON=...), which will "
            f"be recorded in {_log_label()}."
        )
    _append("ALLOWED", context, reason or "(no reason given)")


def log_text() -> str:
    return LOG.read_text() if LOG.exists() else ""
