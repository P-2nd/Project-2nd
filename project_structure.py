from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_FILE = BASE_DIR / "project_structure.txt"

EXCLUDE_DIRS = {
    ".git",
    ".idea",
    "__pycache__",
    ".venv",
    "venv"
}

EXCLUDE_FILES = {
    ".DS_Store"
}


def make_tree(path, prefix=""):

    lines = []

    items = sorted(
        [
            p for p in path.iterdir()
            if p.name not in EXCLUDE_DIRS
            and p.name not in EXCLUDE_FILES
        ],
        key=lambda x: x.name.lower()
    )

    for idx, item in enumerate(items):

        last = idx == len(items) - 1

        branch = "└── " if last else "├── "

        lines.append(
            prefix + branch + item.name
        )

        if item.is_dir():

            next_prefix = (
                prefix + "    "
                if last
                else prefix + "│   "
            )

            lines.extend(
                make_tree(
                    item,
                    next_prefix
                )
            )

    return lines


tree = [
    BASE_DIR.name + "/"
]

tree.extend(
    make_tree(BASE_DIR)
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(tree)
    )

print(
    "생성 완료:",
    OUTPUT_FILE
)