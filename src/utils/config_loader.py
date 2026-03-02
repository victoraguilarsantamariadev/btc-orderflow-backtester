import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file.

    If no path given, looks for config/settings.yaml relative to project root.
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "settings.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_prop_firm_rules(config_path: str = None) -> dict:
    """Load prop firm rules configuration."""
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "prop_firm_rules.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
