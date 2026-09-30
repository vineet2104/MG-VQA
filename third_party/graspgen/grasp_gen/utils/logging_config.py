import logging
import sys

# Global flag to track if logging is initialized
_logging_initialized = False


def setup_logging():
    """
    Set up basic logging configuration for all scripts.
    Uses INFO level and outputs to both console and file.
    """
    global _logging_initialized

    # Only initialize once
    if _logging_initialized:
        return

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    _logging_initialized = True


def get_logger(name):
    """
    Get a logger instance for a specific module.

    Args:
        name: The name of the module (usually __name__)

    Returns:
        A logger instance configured for the module
    """
    # Ensure logging is initialized before getting a logger
    setup_logging()
    return logging.getLogger(name)


# Initialize logging when this module is imported
setup_logging()
