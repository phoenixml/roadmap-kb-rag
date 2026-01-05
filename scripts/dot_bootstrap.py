from pathlib import Path
import shutil


def ensure_defence_dot_inherits_attack_dot(
    attack_dot: str = "TrackA.dot",
    defence_dot: str = "TrackA_with_defences.dot",
):
    attack_dot = Path(attack_dot)
    defence_dot = Path(defence_dot)

    if defence_dot.exists():
        return

    if not attack_dot.exists():
        raise FileNotFoundError(f"Attack DOT not found: {attack_dot}")

    shutil.copy(attack_dot, defence_dot)
