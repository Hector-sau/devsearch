"""Expected, user-reportable failures. CLIs map these to a nonzero exit status."""


class DevSearchError(Exception):
    """Base class for failures that are reported without a stack trace."""


class ConfigError(DevSearchError):
    """A configuration file is missing, malformed, has unknown fields or bad values."""


class ManifestError(DevSearchError):
    """An artifact has an invalid schema, hash, reference or escaping path."""


class InvalidRequestError(DevSearchError):
    """A search request is invalid (HTTP 422 `invalid_request` in docs/04)."""

    code = "invalid_request"
    retryable = False


class BusyError(DevSearchError):
    """The single search slot is occupied (HTTP 429 `busy`); there is no waiting queue."""

    code = "busy"
    retryable = True


class DeadlineExceededError(DevSearchError):
    """The deadline passed before a complete baseline result existed (HTTP 504 `deadline_exceeded`)."""

    code = "deadline_exceeded"
    retryable = True


class IndexUnavailableError(DevSearchError):
    """The index or release cannot be used; no result is fabricated (HTTP 503 `index_unavailable`)."""

    code = "index_unavailable"
    retryable = False



class DeadlineExceeded(DevSearchError):
    """A deadline or stop request arrived between scoring batches; every partial score is discarded."""


class NonFiniteScoreError(DevSearchError):
    """The model produced NaN or infinity; no ranking may be published from it."""


class PrerequisiteError(DevSearchError):
    """A required resource is unavailable: mount, device, asset, free space, lease or deadline (campaign exit 3)."""


class ExecutionFailure(DevSearchError):
    """Execution failed in a way that must not be retried silently: OOM, non-finite values, device loss (exit 4)."""


class Interrupted(DevSearchError):
    """The work was interrupted; resumable state is intact (exit 4, retryable under --resume)."""


class BudgetStop(DevSearchError):
    """The remaining window cannot hold the next unit of work; stop safely and keep resumable state (exit 6)."""


class AwaitingReview(DevSearchError):
    """Post-rank review is outstanding; the campaign pauses without a final report (exit 5)."""
