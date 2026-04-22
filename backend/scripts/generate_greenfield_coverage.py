#!/usr/bin/env python3
"""
Generate backend/postgres_init/GREENFIELD_COVERAGE.md: classify each migrations/*.sql
for greenfield vs 01_init.sql and numbered wrappers (02-09).

Categories:
  A — Objects appear merged in 01_init.sql (heuristic: CREATE TABLE name match)
  B — File is \\ir'd from a top-level 0N_*.sql wrapper (02-09)
  C — Brownfield-only: destructive/rename/seed-only (do not auto-run on greenfield)
  D — Likely gap: CREATE TABLE for a relation not found in 01 (needs merge or wrapper)
  R — Review: no CREATE TABLE parsed (often RLS-only); manual A vs C

Run from repo root or backend/:  python scripts/generate_greenfield_coverage.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POSTGRES_INIT = REPO / "backend" / "postgres_init"
MIGRATIONS = POSTGRES_INIT / "migrations"
INIT_SQL = POSTGRES_INIT / "01_init.sql"
OUT_MD = POSTGRES_INIT / "GREENFIELD_COVERAGE.md"

CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:public\.)?([a-z][a-z0-9_]*)\s*\(",
    re.IGNORECASE | re.MULTILINE,
)

IR_RE = re.compile(r"\\ir\s+migrations/(\S+\.sql)", re.IGNORECASE)

# Never run on greenfield init (brownfield / destructive / superseded by 01 layout)
BROWNFIELD_MARKERS = [
    (re.compile(r"DROP\s+TABLE\s+IF\s+EXISTS\s+agent_skills\b", re.I), "049-style drops agent_skills"),
    (re.compile(r"DROP\s+TABLE\s+agent_skills\b", re.I), "drops agent_skills"),
    (re.compile(r"\bRENAME\s+TO\s+agent_lines\b", re.I), "101-style team→line rename"),
    (re.compile(r"\bRENAME\s+TO\s+agent_line_", re.I), "101-style rename chain"),
    (re.compile(r"ALTER\s+TABLE\s+agent_teams\s+RENAME\b", re.I), "101 renames agent_teams"),
    (re.compile(r"DROP\s+TABLE\s+IF\s+EXISTS\s+agent_service_bindings\b", re.I), "125 drops bindings"),
]

BROWNFIELD_FILENAMES = frozenset(
    {
        "049_agent_factory_ux_cleanup.sql",
        "050_add_agent_service_bindings.sql",
        "101_agent_teams_to_agent_lines.sql",
        "125_drop_agent_service_bindings.sql",
        "075_drop_entertainment_sync.sql",
        "117_drop_news_articles.sql",
        "124_drop_legacy_github_tables.sql",
        # Legacy agent_teams / team_workspace chain — greenfield uses agent_lines + 156 instead
        "047_add_agent_team_watches.sql",
        "086_agent_teams.sql",
        "088_agent_team_goals.sql",
        "089_agent_tasks.sql",  # brownfield name team_id; 01 has agent_tasks with line_id
        "094_team_workspace.sql",
        "100_fix_agent_team_watches_fk.sql",
        "092_team_budget_and_member_color.sql",
        "093_agent_teams_handle.sql",
        "096_team_tool_skills.sql",
        "087_agent_messages.sql",  # team_id-era; 01 has agent_messages with line_id
        "090_team_heartbeat.sql",
    }
)


def load_init_sql() -> str:
    if not INIT_SQL.is_file():
        raise SystemExit(f"Missing {INIT_SQL}")
    return INIT_SQL.read_text(encoding="utf-8", errors="replace")


def collect_wrapper_ir() -> set[str]:
    found: set[str] = set()
    for path in sorted(POSTGRES_INIT.glob("[0-9][0-9]_*.sql")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in IR_RE.finditer(text):
            found.add(m.group(1))
    return found


def table_in_init(table: str, init_text: str) -> bool:
    """True if 01 appears to define this table."""
    pat = re.compile(
        rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:public\.)?{re.escape(table)}\b",
        re.IGNORECASE,
    )
    return bool(pat.search(init_text))


def classify_file(path: Path, init_text: str, wrapper_ir: set[str]) -> tuple[str, str]:
    name = path.name
    body = path.read_text(encoding="utf-8", errors="replace")

    if name in wrapper_ir:
        return "B", "included via \\ir from 02-09 wrapper"

    if name in BROWNFIELD_FILENAMES:
        return "C", "listed brownfield-only filename"

    for rx, why in BROWNFIELD_MARKERS:
        if rx.search(body):
            return "C", f"brownfield marker: {why}"

    creates = CREATE_TABLE_RE.findall(body)
    if not creates:
        if re.search(r"CREATE\s+(UNIQUE\s+)?INDEX", body, re.I):
            return "R", "indexes only / no CREATE TABLE — check if merged in 01"
        if re.search(r"ALTER\s+TABLE.*(?:POLICY|ENABLE\s+ROW\s+LEVEL)", body, re.I):
            return "R", "RLS/ALTER only — likely superseded by 01 policies"
        if re.search(r"^UPDATE\s+", body, re.I | re.M):
            return "R", "UPDATE/seed — brownfield data fix unless seed required"
        return "R", "no CREATE TABLE parsed — manual review"

    missing = [t for t in creates if not table_in_init(t, init_text)]
    if missing:
        return "D", f"CREATE TABLE not found in 01: {', '.join(sorted(set(missing)))}"

    return "A", f"tables in 01: {', '.join(sorted(set(creates)))}"


def main() -> int:
    init_text = load_init_sql()
    wrapper_ir = collect_wrapper_ir()

    rows: list[tuple[str, str, str]] = []
    for path in sorted(MIGRATIONS.glob("*.sql")):
        cat, note = classify_file(path, init_text, wrapper_ir)
        rows.append((path.name, cat, note))

    counts: dict[str, int] = {}
    for _, cat, _ in rows:
        counts[cat] = counts.get(cat, 0) + 1

    lines = [
        "# Greenfield coverage matrix (generated)",
        "",
        "Regenerate: `python backend/scripts/generate_greenfield_coverage.py`",
        "",
        "## Legend",
        "",
        "| Cat | Meaning |",
        "|-----|---------|",
        "| **A** | Heuristic: all `CREATE TABLE` names in this file match a `CREATE TABLE` in `01_init.sql`. |",
        "| **B** | Pulled by `\\ir` from a numbered wrapper `02`–`09`. |",
        "| **C** | Brownfield-only or destructive — **do not** add to greenfield Docker init. |",
        "| **D** | Gap: `CREATE TABLE` for at least one relation **not** found in `01` — needs merge or wrapper. |",
        "| **R** | Review: no table create parsed (RLS/index/UPDATE/DROP-only, etc.). |",
        "",
        "## Summary",
        "",
    ]
    for k in sorted(counts):
        lines.append(f"- **{k}**: {counts[k]}")
    lines.extend(["", "## Per-file classification", "", "| migration | cat | notes |", "|---|---:|---|"])
    for name, cat, note in rows:
        safe = note.replace("|", "\\|")
        lines.append(f"| `{name}` | {cat} | {safe} |")

    lines.extend(
        [
            "",
            "## Wrapper `\\ir` targets (02-09)",
            "",
        ]
    )
    for t in sorted(wrapper_ir):
        lines.append(f"- `{t}`")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD} ({len(rows)} migrations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
