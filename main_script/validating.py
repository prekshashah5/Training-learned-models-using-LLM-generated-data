import sqlglot
from sqlglot.errors import ParseError
from pathlib import Path
import os
import json
from dotenv import load_dotenv
from utils import read_json_file, get_latest_json_path

load_dotenv()

output_folder = Path(os.getenv("OUTPUT_FOLDER", "../output"))
latest_json_path = get_latest_json_path(output_folder)

data = read_json_file(latest_json_path)

valid_data = []
invalid_count = 0
skipped = 0

for item in data:
    if "valid" in item:
        valid_data.append(item)
        skipped += 1
        continue

    sql = item.get("sql")

    try:
        sqlglot.parse_one(sql, read="postgres")
        item["valid"] = True
        item["error"] = None
        valid_data.append(item)
    except ParseError as e:
        invalid_count += 1
with open(latest_json_path, "w") as f:
    json.dump(valid_data, f, indent=2)

print(f"[done] Removed {invalid_count} invalid queries")
print(f"[info] Remaining valid queries: {len(valid_data)}")
print(f"[info] Updated file saved to {latest_json_path}")
print(f"[info] Skipped already-validated queries: {skipped}")