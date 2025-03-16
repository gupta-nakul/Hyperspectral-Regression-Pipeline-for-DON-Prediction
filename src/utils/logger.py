import logging
import sys

def get_logger(name: str = None) -> logging.Logger:
    """
    Creates and returns a logger with the specified name.
    The logger prints messages to stdout with a basic format.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # you can change this to INFO, WARNING, etc.

    # If the logger already has handlers, don't add another
    if not logger.handlers:
        # Create a console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        # Define a simple format
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

        # Optional: If you don't want logs from other libraries, 
        # you can set propagate to False
        logger.propagate = False

    return logger
