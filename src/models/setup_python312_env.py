#!/usr/bin/env python3
"""Create the project virtual environment with Python 3.12 and install dependencies.

Usage:
    python setup_python312_env.py
    python setup_python312_env.py --recreate
    python setup_python312_env.py --python /path/to/python3.12

Set PYTHON312_EXECUTABLE when python3.12 is not on PATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PYTHON_VERSION = (3, 12)

# PyPI 배포판 이름과 import 이름. requirements.txt의 모든 직접 의존성을 점검한다.
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "PyYAML": "yaml",
    "pyarrow": "pyarrow",
    "scikit-learn": "sklearn",
    "imbalanced-learn": "imblearn",
    "joblib": "joblib",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "torch": "torch",
    "tensorboard": "tensorboard",
    "tqdm": "tqdm",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "shap": "shap",
    "optuna": "optuna",
    "streamlit": "streamlit",
    "pytest": "pytest",
}


def find_project_root(start: Path) -> Path:
    """Return the closest parent that contains requirements.txt."""
    for directory in (start, *start.parents):
        if (directory / "requirements.txt").is_file():
            return directory
    raise FileNotFoundError("requirements.txt를 포함한 프로젝트 루트를 찾을 수 없습니다.")


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def python_version(python: Path | str) -> tuple[int, int]:
    result = subprocess.run(
        [str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        check=True,
        capture_output=True,
        text=True,
    )
    major, minor = result.stdout.strip().split(".")
    return int(major), int(minor)


def resolve_python312(requested: str | None) -> str:
    candidates = [
        requested,
        os.environ.get("PYTHON312_EXECUTABLE"),
        shutil.which("python3.12"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if python_version(candidate) == PYTHON_VERSION:
                return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError(
        "Python 3.12 실행 파일을 찾지 못했습니다. Python 3.12를 설치한 뒤 "
        "--python /경로/python3.12 또는 PYTHON312_EXECUTABLE을 지정하세요."
    )


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def verify_environment(python: Path) -> None:
    """Verify the created virtual environment and every direct dependency."""
    if python_version(python) != PYTHON_VERSION:
        raise RuntimeError(f"가상환경 Python 버전이 3.12가 아닙니다: {python_version(python)}")

    print("\n[환경 검증]")
    print(f"- Python: {'.'.join(map(str, python_version(python)))}")
    run([str(python), "-m", "pip", "check"])

    # 이 스크립트 자체는 시스템 Python에서 실행되므로, 검증은 .venv Python으로 다시 실행한다.
    package_check = (
        "from importlib import import_module; "
        "from importlib.metadata import version; "
        f"packages = {REQUIRED_PACKAGES!r}; "
        "print('[패키지 및 import 검증]'); "
        "[(print(f'- {package}: {version(package)}'), import_module(module)) "
        "for package, module in packages.items()]; "
        "import numpy as np; import torch; torch.from_numpy(np.zeros(1)); "
        "print('- torch와 NumPy 연동: 성공'); "
        "print('모든 패키지 검증을 통과했습니다.')"
    )
    try:
        run([str(python), "-c", package_check])
    except subprocess.CalledProcessError as error:
        raise RuntimeError("설치된 패키지 중 import할 수 없는 항목이 있습니다.") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Python 3.12 가상환경과 프로젝트 의존성을 설치합니다.")
    parser.add_argument("--python", help="사용할 Python 3.12 실행 파일 경로")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="기존 .venv를 삭제하고 다시 만듭니다.",
    )
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve().parent)
    requirements = project_root / "requirements.txt"
    environment = project_root / ".venv"

    if environment.exists():
        environment_python = venv_python(environment)
        if environment_python.exists() and python_version(environment_python) == PYTHON_VERSION:
            print(f"기존 Python 3.12 가상환경을 사용합니다: {environment}")
        elif not args.recreate:
            raise RuntimeError(
                f"기존 가상환경({environment})이 Python 3.12가 아닙니다. "
                "교체하려면 --recreate 옵션을 사용하세요."
            )
        else:
            print(f"기존 가상환경을 삭제합니다: {environment}")
            shutil.rmtree(environment)

    if not environment.exists():
        python312 = resolve_python312(args.python)
        print(f"Python 3.12로 가상환경을 만듭니다: {python312}")
        run([python312, "-m", "venv", str(environment)])

    environment_python = venv_python(environment)
    run([str(environment_python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(environment_python), "-m", "pip", "install", "-r", str(requirements)])
    verify_environment(environment_python)

    print("\n설치가 완료되었습니다.")
    if os.name == "nt":
        print(r"활성화: .venv\Scripts\activate")
    else:
        print("활성화: source .venv/bin/activate")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"오류: {error}", file=sys.stderr)
        sys.exit(1)
