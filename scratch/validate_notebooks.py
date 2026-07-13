import ast
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def validate_notebook(nb_path: Path) -> bool:
    print(f"Auditing notebook: {nb_path.name}")
    try:
        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"  [FAIL] Failed to parse JSON: {e}")
        return False

    cells = nb.get("cells", [])
    code_cells_count = 0
    errors = 0

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            code_cells_count += 1
            source = "".join(cell.get("source", []))

            # Syntax validation
            try:
                ast.parse(source)
            except SyntaxError as e:
                print(f"  [FAIL] Cell {idx} contains syntax error: {e}")
                errors += 1
                continue

            # Check imports statically
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name.split(".")[0]
                        try:
                            __import__(name)
                        except ImportError as e:
                            print(f"  [WARN] Cell {idx}: import '{alias.name}' might fail: {e}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Only check if it's a top-level module we can import
                        top_module = node.module.split(".")[0]
                        try:
                            __import__(top_module)
                        except ImportError as e:
                            # Handle relative/internal imports check
                            if node.module.startswith("src"):
                                try:
                                    # Try importing exact path
                                    mod_name = node.module
                                    __import__(mod_name)
                                except ImportError as ie:
                                    print(
                                        f"  [FAIL] Cell {idx}: internal import '{node.module}' failed: {ie}"
                                    )
                                    errors += 1
                            else:
                                print(
                                    f"  [WARN] Cell {idx}: external import '{node.module}' might fail: {e}"
                                )

    if errors == 0:
        print(f"  [OK] Syntax and imports are valid ({code_cells_count} code cells verified).")
        return True
    else:
        print(f"  [FAIL] Completed with {errors} errors.")
        return False


def main():
    notebook_dir = Path(__file__).resolve().parents[1] / "notebooks"
    notebooks = sorted(list(notebook_dir.glob("*.ipynb")))

    print("========================================")
    print("[*] SUPPORTAI NOTEBOOK STATIC VALIDATION")
    print("========================================")

    all_passed = True
    for nb in notebooks:
        passed = validate_notebook(nb)
        if not passed:
            all_passed = False

    print("========================================")
    if all_passed:
        print("[SUCCESS] ALL NOTEBOOKS PASSED STATIC AUDIT!")
        sys.exit(0)
    else:
        print("[FAIL] SOME NOTEBOOKS HAVE ERRORS!")
        sys.exit(1)


if __name__ == "__main__":
    main()
