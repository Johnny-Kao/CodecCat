"""Lightweight community-health verification for CodecCat docs and templates.

Tests local Markdown navigation and GitHub Issue form structure. No native
build, model materialization or CI artifact costs are required.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXTRA = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "MAINTAINING.md",
    ROOT / "ROADMAP.md",
    ROOT / "SECURITY.md",
    ROOT / "CODE_OF_CONDUCT.md",
    ROOT / "SUPPORT.md",
    ROOT / "RELEASE_CHECKLIST_NATIVE.md",
    ROOT / "docs/STABILITY.md",
    ROOT / "docs/PYPI_PUBLISHING.md",
    ROOT / "docs/MODEL_SUBMISSIONS.md",
]
RELEASE_DIR = ROOT / "docs/releases"
release_notes = sorted(RELEASE_DIR.glob("v*.md"))
assert len(release_notes) == 5, ("expected archived v0.1.0 release notes", release_notes)
assert not list(ROOT.glob("RELEASE_NOTES_*.md")), "release notes must stay under docs/releases/"
files = EXTRA + [RELEASE_DIR / "README.md"] + release_notes

link_pattern = re.compile(r"!?\[[^]\n]*\]\(([^)\n]+)\)")
missing = []
for file in files:
    assert file.is_file(), f"missing community file: {file}"
    content = file.read_text(encoding="utf-8")
    assert content.strip(), f"empty document: {file}"
    for match in link_pattern.finditer(content):
        raw = match.group(1).strip()
        # Markdown supports an optional title; all our local links use the simple path syntax.
        dest = raw.split(" ", 1)[0].strip("<>")
        if dest.startswith(("https://", "http://", "mailto:", "#", "data:")):
            continue
        target = unquote(dest.split("#", 1)[0])
        if not target:
            continue
        path = file.parent / target
        if not path.exists():
            missing.append((str(file.relative_to(ROOT)), dest))
assert not missing, f"broken local Markdown links: {missing}"
print("COMMUNITY_MARKDOWN_LINKS_PASS", len(files), "documents")

forms = sorted((ROOT / ".github/ISSUE_TEMPLATE").glob("*.yml"))
assert {p.name for p in forms} == {
    "config.yml", "01-bug.yml", "02-feature.yml",
    "03-performance.yml", "04-question.yml",
    "05-model.yml"
}
for path in forms:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), path
    if path.name == "config.yml":
        assert parsed.get("blank_issues_enabled") is False
        continue
    assert isinstance(parsed.get("name"), str) and parsed["name"]
    assert isinstance(parsed.get("description"), str) and parsed["description"]
    assert isinstance(parsed.get("body"), list) and parsed["body"]
    seen = set()
    for field in parsed["body"]:
        assert field.get("type") in {
            "markdown", "input", "textarea", "dropdown", "checkboxes"
        }, (path, field)
        if field["type"] == "markdown":
            continue
        fid = field["id"]
        assert fid not in seen, (path, fid)
        seen.add(fid)
        assert field.get("attributes", {}).get("label"), (path, fid)
print("COMMUNITY_ISSUE_FORMS_PASS", len(forms) - 1, "forms")

assert (ROOT / ".github/PULL_REQUEST_TEMPLATE.md").is_file()
owners = (ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")
assert "* @Johnny-Kao" in owners
print("COMMUNITY_REVIEW_ROUTING_PASS")
