"""Perception tools. Import concrete modules for new integrations.

Legacy package-level names are resolved lazily to avoid import cycles while
older callers still import shared utilities from tools.py.
"""

def __getattr__(name):
    if name.startswith('__'):
        raise AttributeError(name)
    import tools
    try:
        return getattr(tools, name)
    except AttributeError:
        raise AttributeError(name) from None
