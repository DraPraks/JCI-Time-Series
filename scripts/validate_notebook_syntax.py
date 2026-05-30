import json
import sys
from pathlib import Path


def normalize_jupyter_source(source: str) -> str:
    normalized = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("pip install "):
            indent = line[: len(line) - len(stripped)]
            normalized.append(f"{indent}pass  # Jupyter-only command\n")
        else:
            normalized.append(line)
    return "".join(normalized)


def main() -> int:
    failed = False
    for raw_path in sys.argv[1:]:
        path = Path(raw_path)
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [
            (index, cell)
            for index, cell in enumerate(notebook["cells"])
            if cell["cell_type"] == "code"
        ]
        errors = []
        for index, cell in code_cells:
            source = normalize_jupyter_source("".join(cell["source"]))
            try:
                compile(source, f"{path.name}:cell-{index}", "exec")
            except SyntaxError as error:
                errors.append((index, error))

        print(f"{path}: {len(code_cells)} code cells, {len(errors)} syntax errors")
        for index, error in errors:
            print(f"  cell {index}: line {error.lineno}: {error.msg}")
        failed = failed or bool(errors)

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
