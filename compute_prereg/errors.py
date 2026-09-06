"""Exception hierarchy. Every message is passed through security.redact before it leaves."""

from __future__ import annotations


class PreregError(Exception):
    """Base class."""


class EstimationError(PreregError):
    pass


class UnsupportedTraceError(EstimationError):
    """FakeTensor could not trace the model (unsupported op / dynamic control flow)."""


class LedgerError(PreregError):
    pass


class LedgerIntegrityError(LedgerError):
    """Signature or hash of a stored record does not verify."""


class NotFound(PreregError):
    pass


class PolicyError(PreregError):
    """An action the current actor / mode is not allowed to take (e.g. agent raising a cap)."""


class ConfirmationError(PreregError):
    """Token missing, expired, consumed, or bound to different parameters."""


class EnvelopeViolation(PreregError):
    """An offer or a running job exceeds the envelope in at least one dimension."""

    def __init__(self, message: str, failures: list[str] | None = None):
        super().__init__(message)
        self.failures = failures or []


class NoOfferFits(PreregError):
    pass


class ProvisionError(PreregError):
    pass


class AdapterUnavailable(PreregError):
    """Adapter package or credential is missing."""
