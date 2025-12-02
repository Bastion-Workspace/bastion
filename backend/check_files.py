#!/usr/bin/env python3
from pathlib import Path
import os

uploads_base = Path(os.getenv("UPLOAD_DIR", "/opt/bastion/uploads"))
team_id = "d746a9c2-2564-414a-b660-912e6913f79c"
team_dir = uploads_base / "Teams" / team_id / "posts"

print(f"Upload base: {uploads_base}")
print(f"Team directory: {team_dir}")
print(f"Directory exists: {team_dir.exists()}")

if team_dir.exists():
    files = list(team_dir.glob("*"))
    print(f"Files found ({len(files)}):")
    for f in files:
        if f.is_file():
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
else:
    print("Directory does not exist!")

