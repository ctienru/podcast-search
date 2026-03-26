"""Pipeline-level exceptions."""


class CacheMissError(RuntimeError):
    """Raised in strict-cache mode when vector cache has missing entries."""
