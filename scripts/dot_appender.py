# dot_appender.py

from pathlib import Path


class DotGraphAppender:
    """
    Appends attack + defense nodes to an existing DOT file
    WITHOUT corrupting the final brace.
    """

    def __init__(self, dot_path: str):
        self.dot_path = Path(dot_path)

    def append_block(self, block: str):
        content = self.dot_path.read_text(encoding="utf-8")

        # Remove final closing brace safely
        stripped = content.rstrip()
        if not stripped.endswith("}"):
            raise ValueError("DOT file does not end with '}'")

        stripped = stripped[:-1].rstrip()

        updated = (
            stripped
            + "\n\n"
            + block.strip()
            + "\n\n}"
        )

        self.dot_path.write_text(updated, encoding="utf-8")

    # -------------------------------
    # High-level helper
    # -------------------------------
    def append_attack_defense(
        self,
        attack_node_id: str,
        attack_label: str,
        defense_node_id: str,
        defense_label: str
    ):
        block = f'''
"{attack_node_id}" [label="{attack_label}", shape=plaintext]
"{defense_node_id}" [label="{defense_label}", shape=plaintext]
"{attack_node_id}" -> "{defense_node_id}" [label="DEFENDED_BY"]
'''
        self.append_block(block)
