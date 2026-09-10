"""Failures the Host can distinguish without knowing worker messages."""


class NervusError(Exception):
    pass


class SessionClosedError(NervusError):
    pass


class SessionStateError(NervusError):
    pass


class CodeExecutionError(NervusError):
    """The entry failed; earlier namespace changes and effects remain."""


class ModelError(NervusError):
    """The model contract failed; this is not a model-correctable code error."""


class CapabilityInitializationError(NervusError):
    """The candidate was not published; cleanup failures are included in the message."""


class CapabilityReleaseError(NervusError):
    """Release was attempted, without rollback or automatic retry."""


class WorkingStateLostError(NervusError):
    """Worker exit was confirmed; its entire working environment was lost."""


class TurnDrainError(NervusError):
    """Internal indication to the supervisor to terminate the uncooperative worker."""


class TurnStoppedError(SessionStateError):
    """Internal control signal: this model loop no longer owns execution."""


class ProcessCleanupError(SessionStateError):
    """Worker exit alone is insufficient: external execution is not confirmed stopped."""
