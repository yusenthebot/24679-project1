"""Helpers for loading Roboflow credentials used by the detector and app."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


DEFAULT_CREDENTIALS_FILE = Path(__file__).resolve().parent.parent / "roboflow_credentials.txt"


def load_roboflow_credentials(path: Optional[Path] = None) -> Tuple[Optional[str], Optional[str]]:
    """Return the Roboflow API key and project name from a simple text file."""

    config_path = Path(path) if path else DEFAULT_CREDENTIALS_FILE
    if not config_path.exists():
        return None, None

    api_key: Optional[str] = None
    project: Optional[str] = None

    for line in config_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "api_key":
            api_key = value
        elif key == "project_name":
            project = value

    return api_key, project
