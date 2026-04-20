"""Microsoft To Do tools via Graph (connection-scoped; use email-type Microsoft connection)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.tools.m365_invoke_common import invoke_m365_graph
from orchestrator.utils.action_io_registry import register_action


def _format_todo_lists_for_llm(lists: List[Dict[str, Any]]) -> str:
    """ToolMessage text: explicit ids and names so the model does not invent list titles."""
    if not lists:
        return "Microsoft To Do: no lists returned. Use list_todo_lists again if the user expects lists."
    lines = [
        "Microsoft To Do lists. For get_todo_tasks, pass the list id exactly as shown after id=.",
        "When answering the user, use only these names and ids—do not guess.",
        "",
    ]
    for i, item in enumerate(lists, 1):
        lid = (item.get("id") or "").strip()
        name = (item.get("display_name") or "(unnamed)").strip()
        wk = (item.get("well_known_list_name") or "").strip()
        extra = f"  [built-in: {wk}]" if wk else ""
        lines.append(f'{i}. "{name}"  id={lid}{extra}')
    return "\n".join(lines)


def _format_todo_tasks_for_llm(tasks: List[Dict[str, Any]], list_id: str) -> str:
    """ToolMessage text: verbatim titles so the model cannot substitute placeholders."""
    lid = (list_id or "").strip()
    lines = [
        "Microsoft To Do tasks from Graph. Answer using only the lines below.",
        "Copy each Title exactly in quotes when you mention a task; do not paraphrase or use placeholders.",
        f"list_id: {lid}",
        f"count: {len(tasks)}",
        "",
    ]
    if not tasks:
        lines.append("No tasks in this list (empty).")
        return "\n".join(lines)
    for i, t in enumerate(tasks, 1):
        title = (t.get("title") or "").strip() or "(no title)"
        tid = (t.get("id") or "").strip()
        status = (t.get("status") or "").strip()
        due = (t.get("due_datetime") or "").strip()
        imp = (t.get("importance") or "").strip()
        body = (t.get("body") or "").strip()
        lines.append(f'{i}. Title: "{title}"')
        lines.append(f"   task_id: {tid}")
        lines.append(f"   status: {status}")
        if due:
            lines.append(f"   due: {due}")
        if imp and imp != "normal":
            lines.append(f"   importance: {imp}")
        if body:
            preview = body if len(body) <= 400 else body[:400] + "…"
            lines.append(f"   notes: {preview}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_todo_mutation(operation: str, success: bool, task_id: str, error: str,
                          title: str = "", list_id: str = "") -> str:
    """ToolMessage text for create/update/delete mutations."""
    if not success:
        return f"To Do {operation} failed: {error or 'unknown error'}"
    parts = [f"To Do {operation} succeeded."]
    if title:
        parts.append(f'  title: "{title}"')
    if task_id:
        parts.append(f"  task_id: {task_id}")
    if list_id:
        parts.append(f"  list_id: {list_id}")
    return "\n".join(parts)


class ConnParams(BaseModel):
    connection_id: Optional[int] = Field(default=None, description="Microsoft 365 connection id")


class ListTodoListsInputs(ConnParams):
    pass


class TodoListsOutputs(BaseModel):
    lists: List[Dict[str, Any]] = Field(default_factory=list)
    formatted: str = Field(description="Human-readable summary")


async def list_todo_lists_tool(
    user_id: str = "system",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph("list_todo_lists", user_id, connection_id, {})
    lists = out.get("lists") or []
    lists = lists if isinstance(lists, list) else []
    err = out.get("error")
    body = _format_todo_lists_for_llm(lists)
    return {
        "lists": lists,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


class GetTodoTasksInputs(ConnParams):
    list_id: str = Field(
        description="List id from list_todo_lists (id field). Do not use display names; "
        "aliases 'flagged' and 'tasks' are accepted and resolved to built-in lists."
    )
    top: int = Field(default=50, ge=1, le=200, description="Maximum tasks to return (1-200)")


class TodoTasksOutputs(BaseModel):
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    formatted: str = Field(description="Human-readable summary")


async def get_todo_tasks_tool(
    user_id: str = "system",
    list_id: str = "",
    top: int = 50,
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "get_todo_tasks",
        user_id,
        connection_id,
        {"list_id": list_id, "top": top},
    )
    tasks = out.get("tasks") or []
    tasks = tasks if isinstance(tasks, list) else []
    resolved_lid = list_id
    if tasks and isinstance(tasks[0], dict) and (tasks[0].get("list_id") or "").strip():
        resolved_lid = str(tasks[0]["list_id"]).strip()
    err = out.get("error")
    body = _format_todo_tasks_for_llm(tasks, resolved_lid)
    return {
        "tasks": tasks,
        "formatted": (f"Error: {err}\n\n" if err else "") + body,
    }


class CreateTodoTaskInputs(ConnParams):
    list_id: str = Field(
        description="List id from list_todo_lists (id field), not the list title"
    )
    title: str = Field(description="Task title")
    body: str = Field(default="", description="Optional notes")
    due_datetime: str = Field(default="", description="ISO 8601 due date/time")
    importance: str = Field(default="normal", description="low, normal, high")


class TodoMutationOutputs(BaseModel):
    task_id: str = Field(default="", description="Created task id when applicable")
    success: bool = Field(default=True)
    formatted: str = Field(description="Human-readable summary")


async def create_todo_task_tool(
    user_id: str = "system",
    list_id: str = "",
    title: str = "Task",
    body: str = "",
    due_datetime: str = "",
    importance: str = "normal",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "create_todo_task",
        user_id,
        connection_id,
        {
            "list_id": list_id,
            "title": title,
            "body": body,
            "due_datetime": due_datetime,
            "importance": importance,
        },
    )
    ok = out.get("success", False)
    tid = out.get("task_id") or ""
    err = out.get("error") or ""
    return {
        "task_id": tid,
        "success": ok,
        "formatted": _format_todo_mutation("create_task", ok, tid, err, title=title, list_id=list_id),
    }


class UpdateTodoTaskInputs(ConnParams):
    list_id: str = Field(
        description="List id from list_todo_lists (the long id string, not the list title)"
    )
    task_id: str = Field(
        description="Task id from get_todo_tasks (the long id string, not the task title)"
    )
    title: Optional[str] = Field(default=None, description="New title (omit to leave unchanged)")
    body: Optional[str] = Field(default=None, description="New notes/body (omit to leave unchanged)")
    status: Optional[str] = Field(
        default=None,
        description="notStarted, inProgress, completed, waitingOnOthers, deferred (omit to leave unchanged)"
    )
    due_datetime: Optional[str] = Field(default=None, description="ISO 8601 due date/time (omit to leave unchanged)")
    importance: Optional[str] = Field(default=None, description="low, normal, high (omit to leave unchanged)")


class TodoOkOutputs(BaseModel):
    success: bool = Field(default=True)
    formatted: str = Field(description="Human-readable summary")


async def update_todo_task_tool(
    user_id: str = "system",
    list_id: str = "",
    task_id: str = "",
    title: Optional[str] = None,
    body: Optional[str] = None,
    status: Optional[str] = None,
    due_datetime: Optional[str] = None,
    importance: Optional[str] = None,
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"list_id": list_id, "task_id": task_id}
    if title is not None:
        params["title"] = title
    if body is not None:
        params["body"] = body
    if status is not None:
        params["status"] = status
    if due_datetime is not None:
        params["due_datetime"] = due_datetime
    if importance is not None:
        params["importance"] = importance
    out = await invoke_m365_graph("update_todo_task", user_id, connection_id, params)
    ok = out.get("success", False)
    err = out.get("error") or ""
    return {
        "success": ok,
        "formatted": _format_todo_mutation("update_task", ok, task_id, err, list_id=list_id),
    }


class DeleteTodoTaskInputs(ConnParams):
    list_id: str = Field(
        description="List id from list_todo_lists (the long id string, not the list title)"
    )
    task_id: str = Field(
        description="Task id from get_todo_tasks (the long id string, not the task title)"
    )


async def delete_todo_task_tool(
    user_id: str = "system",
    list_id: str = "",
    task_id: str = "",
    connection_id: Optional[int] = None,
) -> Dict[str, Any]:
    out = await invoke_m365_graph(
        "delete_todo_task",
        user_id,
        connection_id,
        {"list_id": list_id, "task_id": task_id},
    )
    ok = out.get("success", False)
    err = out.get("error") or ""
    return {
        "success": ok,
        "formatted": _format_todo_mutation("delete_task", ok, task_id, err, list_id=list_id),
    }


register_action(
    name="list_todo_lists",
    category="productivity",
    description="List Microsoft To Do lists for the connected account",
    inputs_model=ListTodoListsInputs,
    outputs_model=TodoListsOutputs,
    tool_function=list_todo_lists_tool,
)
register_action(
    name="get_todo_tasks",
    category="productivity",
    description="List tasks in a Microsoft To Do list. Tool output includes exact titles—repeat them verbatim when replying.",
    inputs_model=GetTodoTasksInputs,
    outputs_model=TodoTasksOutputs,
    tool_function=get_todo_tasks_tool,
)
register_action(
    name="create_todo_task",
    category="productivity",
    description="Create a task in a Microsoft To Do list",
    inputs_model=CreateTodoTaskInputs,
    outputs_model=TodoMutationOutputs,
    tool_function=create_todo_task_tool,
)
register_action(
    name="update_todo_task",
    category="productivity",
    description="Update a Microsoft To Do task",
    inputs_model=UpdateTodoTaskInputs,
    outputs_model=TodoOkOutputs,
    tool_function=update_todo_task_tool,
)
register_action(
    name="delete_todo_task",
    category="productivity",
    description="Delete a Microsoft To Do task",
    inputs_model=DeleteTodoTaskInputs,
    outputs_model=TodoOkOutputs,
    tool_function=delete_todo_task_tool,
)
