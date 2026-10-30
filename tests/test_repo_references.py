"""Every repo path / module / markdown link written in code or docs must exist.

Catches stale references after files move: `scripts/...`, `skills/...`, `mighty_mouse/...`,
`tests/...`, `configs/...`, `docs/...` paths, dotted `mighty_mouse.a.b` module names and
relative markdown links. CHANGELOG.md files are history and are not checked, nor is this file
(its examples are deliberately fake).
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SUFFIXES = {".md", ".py", ".sh", ".yaml", ".yml"}
SKIP_DIRS = {".git", ".idea", ".ruff_cache", ".pytest_cache", ".ipynb_checkpoints", "__pycache__",
             "notebooks"}
# generic example paths used to explain a convention, not real files
EXAMPLES = {"mighty_mouse/a/b.py", "mighty_mouse/<stage>/b.py", "tests/a/test_b.py",
            "tests/<stage>/test_b.py"}

PATH_RE = re.compile(
    r"(?<![\w/.<>{}-])((?:scripts|skills|mighty_mouse|tests|configs|docs)/[^\s`'\"()\[\]|,;:]*)"
)
MODULE_RE = re.compile(r"(?<![\w/.])(mighty_mouse(?:\.[a-z_][a-z_0-9]*)+)")
LINK_RE = re.compile(r"\]\(([^)\s#]+)(?:#[^)\s]*)?\)")


def source_files() -> list[Path]:
    return sorted(
        p for p in REPO.rglob("*")
        if p.is_file() and p.suffix in SUFFIXES and p.name != "CHANGELOG.md"
        and not SKIP_DIRS & set(p.relative_to(REPO).parts) and p != Path(__file__).resolve()
    )


def is_placeholder(token: str) -> bool:
    return any(c in token for c in "<>{}*") or "..." in token or "$" in token


def missing_path(token: str) -> bool:
    token = token.rstrip(".:")
    if is_placeholder(token) or token in EXAMPLES:
        return False
    return not (REPO / token).exists()


def missing_module(dotted: str) -> bool:
    """True if the first segment after `mighty_mouse` is not a module / package."""
    parts = dotted.split(".")[1:]
    base = REPO / "mighty_mouse"
    return not ((base / parts[0]).is_dir() or (base / f"{parts[0]}.py").exists())


def find_stale(path: Path) -> list[str]:
    stale = []
    rel = path.relative_to(REPO)
    for n, line in enumerate(path.read_text().splitlines(), 1):
        for m in PATH_RE.finditer(line):
            if missing_path(m[1]):
                stale.append(f"{rel}:{n}: {m[1]}")
        for m in MODULE_RE.finditer(line):
            if missing_module(m[1]):
                stale.append(f"{rel}:{n}: {m[1]}")
        if path.suffix == ".md":
            for m in LINK_RE.finditer(line):
                target = m[1]
                if "://" in target or target.startswith("mailto:") or is_placeholder(target):
                    continue
                if not (path.parent / target).resolve().exists():
                    stale.append(f"{rel}:{n}: link {target}")
    return stale


class TestFindStale:
    """Test the repo-wide reference check."""

    def test_find_stale_no_dead_references(self):
        stale = [s for p in source_files() for s in find_stale(p)]
        assert not stale, "dead references:\n" + "\n".join(stale)

    def test_find_stale_flags_missing_path(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("# see scripts/does_not_exist.py\n")
        with __import__("unittest.mock").mock.patch(f"{__name__}.REPO", tmp_path):
            assert find_stale(f) == ["x.py:1: scripts/does_not_exist.py"]

    def test_find_stale_flags_missing_module(self):
        assert missing_module("mighty_mouse.pretraining.dedup")
        assert not missing_module("mighty_mouse.pretrain.dedup")
        assert not missing_module("mighty_mouse.paths.load_paths")

    def test_find_stale_ignores_placeholders(self):
        assert not missing_path("scripts/preprocess/<dataset>/")
        assert not missing_path("tests/<stage>/test_b.py")
