import json
import os

_ENTITIES_NAMES = None
_NAMES_ENTITIES = None
_ENTITIES_PATH = None


def resolve_entities_names_path(path=None):
    candidates = []
    if path:
        candidates.append(path)
    env_path = os.getenv("ENTITIES_NAMES_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(os.getcwd(), "entities_names.json"))
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.append(os.path.join(base, "entities_names.json"))
    repo_root = os.path.abspath(os.path.join(base, ".."))
    candidates.append(os.path.join(repo_root, "entities_names.json"))

    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "entities_names.json not found. Provide --entities_names_path or set ENTITIES_NAMES_PATH."
    )


def set_entities_names_path(path):
    global _ENTITIES_PATH, _ENTITIES_NAMES, _NAMES_ENTITIES
    _ENTITIES_PATH = path
    _ENTITIES_NAMES = None
    _NAMES_ENTITIES = None


def get_entities_names(path=None):
    global _ENTITIES_NAMES
    if _ENTITIES_NAMES is not None:
        return _ENTITIES_NAMES
    load_path = _ENTITIES_PATH or path
    load_path = resolve_entities_names_path(load_path)
    with open(load_path, "r") as f:
        _ENTITIES_NAMES = json.load(f)
    return _ENTITIES_NAMES


def get_names_entities(path=None):
    global _NAMES_ENTITIES
    if _NAMES_ENTITIES is not None:
        return _NAMES_ENTITIES
    ents = get_entities_names(path=path)
    _NAMES_ENTITIES = {v: k for k, v in ents.items()}
    return _NAMES_ENTITIES
