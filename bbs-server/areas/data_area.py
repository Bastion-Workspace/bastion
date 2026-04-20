"""
Data workspace browser: workspaces, databases, tables, SQL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from rendering.paginator import paginate_text
from rendering.tables import render_table
from rendering.text import format_header_context, section_header, word_wrap

if TYPE_CHECKING:
    from session import BBSSession


async def data_explorer(session: "BBSSession") -> None:
    await session.clear_screen()
    client = session.client
    jwt = session.jwt_token
    t = session.theme
    hdr = section_header(
        "Data Workspace",
        session.term_width - 2,
        t,
        context=format_header_context(session.display_name or session.username),
    )
    await session._write(f"\r\n{hdr}\r\n")

    workspaces = await client.list_workspaces(jwt)
    if not workspaces:
        await session._write("\r\nNo data workspaces.\r\n")
        return

    while True:
        await session._write("\r\nWorkspaces:\r\n")
        for i, w in enumerate(workspaces, 1):
            wid = w.get("workspace_id") or w.get("id", "")
            name = (w.get("name") or wid)[:50]
            await session._write(f"  {i}) {name}\r\n")
        await session._write(
            f"\r\n{t.fg_bright_green}[#]{t.reset} select  "
            f"{t.fg_bright_green}[B]{t.reset}ack: "
        )
        choice = (await session.read_line()).strip().lower()
        if choice in ("b", "back", "q"):
            return
        if not choice.isdigit():
            continue
        n = int(choice)
        if n < 1 or n > len(workspaces):
            continue
        ws = workspaces[n - 1]
        ws_id = ws.get("workspace_id") or ws.get("id")
        if not ws_id:
            continue
        await _workspace_menu(session, str(ws_id))


async def _workspace_menu(session: "BBSSession", workspace_id: str) -> None:
    client = session.client
    jwt = session.jwt_token
    t = session.theme

    while True:
        dbs = await client.list_databases(jwt, workspace_id)
        await session._write("\r\nDatabases:\r\n")
        if not dbs:
            await session._write("  (none)\r\n")
        db_list: List[Dict[str, Any]] = list(dbs) if dbs else []
        for i, d in enumerate(db_list, 1):
            did = d.get("database_id") or d.get("id", "")
            name = (d.get("name") or did)[:45]
            await session._write(f"  {i}) {name}\r\n")
        await session._write(
            f"\r\n{t.fg_bright_green}[#]{t.reset} open database  "
            f"{t.fg_bright_green}[Q]{t.reset}uery SQL  "
            f"{t.fg_bright_green}[B]{t.reset}ack: "
        )
        choice = (await session.read_line()).strip().lower()
        if choice in ("b", "back"):
            return
        if choice == "q":
            await _sql_prompt(session, workspace_id)
            continue
        if not choice.isdigit():
            continue
        n = int(choice)
        if n < 1 or n > len(db_list):
            continue
        db = db_list[n - 1]
        db_id = db.get("database_id") or db.get("id")
        if not db_id:
            continue
        await _tables_menu(session, workspace_id, str(db_id))


async def _tables_menu(session: "BBSSession", workspace_id: str, database_id: str) -> None:
    client = session.client
    jwt = session.jwt_token
    t = session.theme
    tables = await client.list_tables(jwt, database_id)

    while True:
        await session._write("\r\nTables:\r\n")
        if not tables:
            await session._write("  (none)\r\n")
        for i, tb in enumerate(tables, 1):
            tid = tb.get("table_id") or tb.get("id", "")
            name = (tb.get("name") or tid)[:45]
            await session._write(f"  {i}) {name}\r\n")
        await session._write(
            f"\r\n{t.fg_bright_green}[#]{t.reset} view data  "
            f"{t.fg_bright_green}[B]{t.reset}ack: "
        )
        choice = (await session.read_line()).strip().lower()
        if choice in ("b", "back"):
            return
        if not choice.isdigit():
            continue
        n = int(choice)
        if n < 1 or n > len(tables):
            continue
        tid = tables[n - 1].get("table_id") or tables[n - 1].get("id")
        if not tid:
            continue
        data = await client.get_table_data(jwt, str(tid), limit=40)
        if data.get("error"):
            await session._write(f"Error: {data['error'][:200]}\r\n")
            continue
        rows_raw = data.get("rows") or []
        lines_out: List[str] = []
        for rd in rows_raw:
            if isinstance(rd, dict):
                row_data = rd.get("row_data") or rd
                if isinstance(row_data, dict):
                    for k, v in list(row_data.items())[:12]:
                        lines_out.append(f"  {k}: {v}")
            lines_out.append("  ---")
        text = "\n".join(lines_out) if lines_out else "(empty)"
        wlines = word_wrap(text, session.term_width - 2)
        page_h = max(5, session.term_height - 3)

        async def wl(s: str) -> None:
            await session._write(s)

        await paginate_text(
            wlines,
            page_h,
            t,
            wl,
            session.read_pager_key,
            after_line_input_drain=session.drain_stray_line_terminators,
        )


async def _sql_prompt(session: "BBSSession", workspace_id: str) -> None:
    await session._write(
        "Enter SQL (single line). SELECT recommended.\r\nSQL> "
    )
    sql = (await session.read_line()).strip()
    if not sql:
        return
    result = await session.client.run_sql_query(session.jwt_token, workspace_id, sql, limit=100)
    if result.get("error"):
        err = result.get("error_message") or result.get("error")
        await session._write(f"Error: {str(err)[:400]}\r\n")
        return
    if result.get("error_message"):
        await session._write(f"SQL error: {result['error_message'][:400]}\r\n")
        return
    cols = result.get("column_names") or []
    recs = result.get("results") or []
    if not cols and recs and isinstance(recs[0], dict):
        cols = list(recs[0].keys())
    rows = []
    for r in recs[:50]:
        if isinstance(r, dict):
            rows.append([str(r.get(c, ""))[:24] for c in cols])
        else:
            rows.append([str(r)])
    if cols:
        w = max(6, min(20, (session.term_width - 10) // max(len(cols), 1)))
        tbl = render_table(cols, rows, col_widths=[w] * len(cols), max_width=session.term_width)
        await session._write("\r\n" + tbl + "\r\n")
    else:
        await session._write(f"\r\n{result.get('result_count', 0)} row(s).\r\n")
