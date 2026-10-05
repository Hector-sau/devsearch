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
