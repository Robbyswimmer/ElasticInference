"""Scaling package."""

__all__ = ["ScalingController"]


def __getattr__(name):
    if name == "ScalingController":
        from scaling.controller import ScalingController
        return ScalingController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
