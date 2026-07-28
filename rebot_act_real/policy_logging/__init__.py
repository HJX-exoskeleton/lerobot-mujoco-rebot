"""Research-grade logging utilities for real-robot policy evaluation.

Recorder imports are lazy so the offline analyzer does not load OpenCV's Qt
plugins before Matplotlib selects its headless rendering backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .recorder import PolicyRunRecorder

__all__ = ["PolicyRunRecorder", "add_recording_arguments"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .recorder import PolicyRunRecorder, add_recording_arguments

        return {
            "PolicyRunRecorder": PolicyRunRecorder,
            "add_recording_arguments": add_recording_arguments,
        }[name]
    raise AttributeError(name)
