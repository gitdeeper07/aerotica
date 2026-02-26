"""AEROTICA custom exceptions."""

class AEROTICAError(Exception):
    """Base exception for all AEROTICA errors."""
    pass


class ConfigurationError(AEROTICAError):
    """Raised when there's a configuration error."""
    pass


class DataError(AEROTICAError):
    """Raised when there's an error with input data."""
    pass


class ParameterError(AEROTICAError):
    """Raised when there's an error with parameter computation."""
    pass


class ModelError(AEROTICAError):
    """Raised when there's an error with model loading or inference."""
    pass


class ValidationError(AEROTICAError):
    """Raised when validation fails."""
    pass


class FileNotFoundError(AEROTICAError):
    """Raised when a required file is not found."""
    pass


class InvalidFormatError(AEROTICAError):
    """Raised when file format is invalid."""
    pass


class ComputationError(AEROTICAError):
    """Raised when computation fails."""
    pass


class ConvergenceError(AEROTICAError):
    """Raised when optimization fails to converge."""
    pass


class OutOfBoundsError(AEROTICAError):
    """Raised when values are out of valid bounds."""
    pass


class MissingParameterError(AEROTICAError):
    """Raised when required parameters are missing."""
    pass


class DatabaseError(AEROTICAError):
    """Raised when there's a database error."""
    pass


class APIError(AEROTICAError):
    """Raised when there's an API error."""
    pass


class AuthenticationError(AEROTICAError):
    """Raised when authentication fails."""
    pass


class RateLimitError(AEROTICAError):
    """Raised when rate limit is exceeded."""
    pass


def handle_exception(func):
    """Decorator for handling exceptions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AEROTICAError as e:
            # Log and re-raise
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"AEROTICA error: {e}")
            raise
        except Exception as e:
            # Wrap other exceptions
            raise AEROTICAError(f"Unexpected error: {e}") from e
    return wrapper
