import os
import yaml
from dotenv import load_dotenv

load_dotenv()

def _expand_env(value: str) -> str:
    if not isinstance(value, str):
        return value
    # naive ${VAR} substitution
    out = value
    for _ in range(10):
        start = out.find("${")
        if start == -1:
            break
        end = out.find("}", start)
        if end == -1:
            break
        key = out[start+2:end]
        out = out[:start] + (os.getenv(key, "") or "") + out[end+1:]
    return out

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k,v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return _expand_env(x)
    return walk(data)
