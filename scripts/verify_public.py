"""Verify public-only wheel/sdist and tests in a clean environment; no model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from export_public import FORBIDDEN, export_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("dist/public-0.1.0a1"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required")
    interpreter = Path(sys.executable).resolve()
    report = {"checks": [], "real_model_calls": 0, "scope": "public source only"}
    with tempfile.TemporaryDirectory(prefix="nervus-public-") as temporary:
        clean = Path(temporary)
        source = export_tree(root, clean / "source")
        for name in ("home", "cache", "config", "tmp", "neutral"):
            (clean / name).mkdir()
        env = {
            "PATH": os.pathsep.join(dict.fromkeys((str(Path(uv).parent), str(interpreter.parent), "/usr/bin", "/bin"))),
            "HOME": str(clean / "home"), "XDG_CONFIG_HOME": str(clean / "config"),
            "XDG_CACHE_HOME": str(clean / "cache"), "UV_CACHE_DIR": str(clean / "cache" / "uv"),
            "TMPDIR": str(clean / "tmp"), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1", "UV_NO_CONFIG": "1", "UV_PYTHON_DOWNLOADS": "never",
        }

        def run(label, command, cwd=None):
            result = subprocess.run([str(p) for p in command], cwd=source if cwd is None else cwd,
                                    env=env, capture_output=True, text=True, timeout=180)
            report["checks"].append({"name": label, "returncode": result.returncode,
                                     "stdout": result.stdout, "stderr": result.stderr})
            print(label, "PASS" if result.returncode == 0 else "FAIL", flush=True)
            if result.returncode:
                raise RuntimeError(f"{label}: {result.stderr}\n{result.stdout}")

        run("build public-only wheel and sdist", [uv, "build", "--offline", "--python", interpreter,
                                                "--out-dir", clean / "artifacts", source])
        wheel, = (clean / "artifacts").glob("*.whl")
        sdist, = (clean / "artifacts").glob("*.tar.gz")
        with zipfile.ZipFile(wheel) as archive:
            wheel_names = archive.namelist()
            license_name, = (n for n in wheel_names if n.endswith(".dist-info/licenses/LICENSE"))
            assert archive.read(license_name) == (root / "LICENSE").read_bytes()
            metadata_name, = (n for n in wheel_names if n.endswith(".dist-info/METADATA"))
            assert b"License-Expression: MIT" in archive.read(metadata_name)
        with tarfile.open(sdist) as archive:
            sdist_names = archive.getnames()
            for name in wheel_names + sdist_names:
                assert not FORBIDDEN.intersection(Path(name).parts), name
            archive.extractall(clean / "unpacked", filter="data")
        report["archive_members"] = {"wheel": wheel_names, "sdist": sdist_names}
        source, = (clean / "unpacked").iterdir()
        assert (source / "LICENSE").read_bytes() == (root / "LICENSE").read_bytes()
        assert (source / ".env.example").exists()
        run("rebuild from public sdist", [uv, "build", "--default-index", "https://pypi.org/simple",
                                        "--python", interpreter, "--wheel", "--out-dir", clean / "rebuilt", sdist])
        run("create clean environment", [uv, "venv", "--offline", "--python", interpreter, clean / "env"])
        python = clean / "env" / "bin" / "python"
        run("install public wheel", [uv, "pip", "install", "--offline", "--no-index", "--no-deps",
                                     "--python", python, wheel])
        run("installed import", [python, "-I", "-c", "import nervus, pathlib, sys; "
                                 "assert pathlib.Path(nervus.__file__).is_relative_to(sys.prefix); print(nervus.__file__)"],
            cwd=clean / "neutral")
        run("installed CLI help", [clean / "env" / "bin" / "nervus", "--help"], cwd=clean / "neutral")
        rebuilt, = (clean / "rebuilt").glob("*.whl")
        run("install rebuilt wheel", [uv, "pip", "install", "--offline", "--no-index", "--no-deps",
                                      "--force-reinstall", "--python", python, rebuilt])
        shutil.rmtree(source / "src")
        for example in ("basic_session", "scripted_session", "host_session", "partial_update"):
            run("example " + example, [python, source / "examples" / (example + ".py")], cwd=clean / "neutral")
        run("public regression suite", [python, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"])
        report["artifacts"] = {}
        for artifact in (wheel, sdist):
            shutil.copy2(artifact, output / artifact.name)
            report["artifacts"][artifact.name] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        report["status"] = "passed"
    (output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in report["artifacts"].items()))
    print(f"Saved public-only artifacts: {output}")


if __name__ == "__main__":
    main()
