"""Locate the PHINO checkout used by the experiment scripts.

The pinned copy is the ``external/phino`` submodule. ``--phino-root`` overrides
this. ``PHINO_ROOT`` is the environment override when the submodule is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

_MARKER = Path("train") / "data" / "mujoco_engine.py"


def _complete(path: Path) -> bool:
    return (path / _MARKER).is_file()


def default_phino_root() -> Path:
    """Return a PHINO source tree, preferring the repository submodule."""
    bundled = Path(__file__).resolve().parents[1] / "external" / "phino"
    if _complete(bundled):
        return bundled
    env = os.environ.get("PHINO_ROOT")
    if env and _complete(Path(env).expanduser()):
        return Path(env).expanduser()
    # Optional checkout used on the machine where the grids were generated,
    # and only when the submodule has not been initialized.
    legacy = Path("/nbu/people/ashah264/phino")
    if _complete(legacy):
        return legacy
    return bundled
