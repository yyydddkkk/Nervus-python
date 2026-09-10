"""Export a source-only release tree without Git history or internal material."""
import argparse
from pathlib import Path
import shutil

ROOT_FILES = (
    ".gitignore", ".env.example", ".python-version", "README.md", "README.zh-CN.md", "LICENSE",
    "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md", "pyproject.toml", "uv.lock",
)
SCRIPT_FILES = ("scripts/export_public.py", "scripts/verify_public.py")
FORBIDDEN = {"docs", "experiments", "prototype", "CONTEXT.md", ".git", ".env", ".venv"}


def export_tree(root: Path, destination: Path):
    root, destination = root.resolve(), destination.resolve()
    # Do not overwrite a previous release snapshot or source checkout.
    destination.mkdir(parents=True, exist_ok=False)
    selected = [root / name for name in ROOT_FILES + SCRIPT_FILES]
    for directory in ("src", "tests", ".github"):
        selected.extend(p for p in (root / directory).rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts
                        and p.suffix not in {".pyc", ".pyo"})
    selected.extend((root / "examples").glob("*.py"))
    for path in sorted(set(selected)):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported in release inputs: {path.name}")
        relative = path.relative_to(root)
        if any(part in FORBIDDEN for part in relative.parts):
            raise ValueError(f"Internal path in release inputs: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="A directory that does not yet exist")
    args = parser.parse_args()
    destination = export_tree(Path(__file__).resolve().parents[1], args.destination)
    print(f"Public source snapshot: {destination}")
    print("No Git history was copied. No commit, push, or upload was performed.")


if __name__ == "__main__":
    main()
