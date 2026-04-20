"""
Rewrite SQL table references to quoted PostgreSQL identifiers when needed.

Workspace views are created as quoted identifiers (see TableService). Unquoted
names in queries are folded to lowercase, so FROM Equipment becomes equipment
and does not match the stored "Equipment" relation.
"""

import re
from typing import Dict, List


def quote_pg_identifier_if_needed(name: str) -> str:
    """Return a safe SQL table reference: unquoted if PostgreSQL folds to the same name, else quoted."""
    if re.match(r"^[a-z_][a-z0-9_]*$", name) and name == name.lower():
        return name
    return '"' + name.replace('"', '""') + '"'


def _ci_regex_for_word(word: str) -> str:
    """Case-insensitive regex matching a single SQL identifier (ASCII)."""
    return "".join(
        f"[{c.lower()}{c.upper()}]" if c.isalpha() else re.escape(c) for c in word
    )


def rewrite_workspace_table_identifiers_in_sql(
    sql: str,
    lower_to_exact: Dict[str, str],
    referenced_tables_lower: List[str],
) -> str:
    """
    Quote known workspace table names in FROM / JOIN / UPDATE / INSERT INTO / DELETE FROM
    when the catalog name is not all-lowercase unquoted-safe.

    Only touches names appearing in referenced_tables_lower (from the query extractor).
    """
    if not sql or not lower_to_exact or not referenced_tables_lower:
        return sql

    join_kw = r"(?:INNER|LEFT|RIGHT|FULL|CROSS)\s+JOIN|JOIN"
    prefix_pat = (
        rf"(?P<prefix>\b(?:FROM|{join_kw}|UPDATE|INTO|DELETE\s+FROM)\s+)"
        rf"(?P<schema>(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?)"
    )

    out = sql
    # Longer names first to avoid partial replacements
    ordered = sorted(
        {t.lower() for t in referenced_tables_lower},
        key=len,
        reverse=True,
    )
    for name_lower in ordered:
        exact = lower_to_exact.get(name_lower)
        if not exact:
            continue
        quoted = quote_pg_identifier_if_needed(exact)
        if quoted == exact:
            continue
        tbl_re = _ci_regex_for_word(exact)
        pat = re.compile(prefix_pat + rf"(?P<tbl>{tbl_re})\b", re.IGNORECASE)
        out = pat.sub(
            lambda m: m.group("prefix") + m.group("schema") + quoted,
            out,
        )

    out = _rewrite_comma_separated_from_tables(out, lower_to_exact)

    return out


_FROM_END_KW = re.compile(
    r"\b(?:WHERE|GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING|UNION|EXCEPT|INTERSECT|OFFSET|FETCH|FOR\s+UPDATE)\b",
    re.IGNORECASE,
)


def _rewrite_comma_separated_from_tables(
    sql: str,
    lower_to_exact: Dict[str, str],
) -> str:
    """Quote comma-separated tables in FROM ... WHERE (parenthesis-aware)."""
    if not sql:
        return sql

    result_parts: List[str] = []
    pos = 0
    for from_m in re.finditer(r"\bFROM\s+", sql, re.IGNORECASE):
        result_parts.append(sql[pos : from_m.start()])
        start = from_m.end()
        rest = sql[start:]
        end_m = _FROM_END_KW.search(rest)
        seg_end = start + (end_m.start() if end_m else len(rest))
        segment = sql[start:seg_end]
        new_segment = _rewrite_commas_in_from_segment(segment, lower_to_exact)
        result_parts.append(sql[from_m.start() : start] + new_segment)
        pos = seg_end

    result_parts.append(sql[pos:])
    return "".join(result_parts)


def _rewrite_commas_in_from_segment(
    segment: str,
    lower_to_exact: Dict[str, str],
) -> str:
    out: List[str] = []
    i = 0
    depth = 0
    while i < len(segment):
        c = segment[i]
        if c == "(":
            depth += 1
            out.append(c)
            i += 1
        elif c == ")":
            depth = max(0, depth - 1)
            out.append(c)
            i += 1
        elif c == "," and depth == 0:
            m = re.match(
                r",\s*((?:[a-zA-Z_][a-zA-Z0-9_]*\.)?)([a-zA-Z_][a-zA-Z0-9_]*)",
                segment[i:],
                re.IGNORECASE,
            )
            if m:
                schema, raw_name = m.group(1), m.group(2)
                exact = lower_to_exact.get(raw_name.lower())
                if exact is not None:
                    repl = quote_pg_identifier_if_needed(exact)
                    out.append(", " + schema + repl)
                    i += m.end()
                    continue
            out.append(c)
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)
