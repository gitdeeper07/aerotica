"""Version information."""

__version__ = "1.0.0"
VERSION = __version__

VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable"
}

def get_version():
    """Return version string."""
    return __version__

def get_version_info():
    """Return version info dictionary."""
    return VERSION_INFO
