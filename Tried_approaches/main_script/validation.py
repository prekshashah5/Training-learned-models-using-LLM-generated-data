import json
import re

def try_parse_json(text):
    """
    Attempts to parse the text as JSON.
    Returns (parsed_data, True) if parsed, (None, False) otherwise.
    """
    try:
        data = json.loads(text)
        return data, True
    except Exception:
        return None, False



def extract_sql_with_regex(text):
    """
    Extracts SQL queries using regex.
    Works even if JSON is invalid or SQL contains subqueries.
    """
    SQL_SELECT_REGEX = re.compile(
        r"SELECT\b[\s\S]*?(?=(?:\"|\n\s*\}|,\s*\{|\Z))",
        re.IGNORECASE
    )

    matches = SQL_SELECT_REGEX.findall(text)  # returns list of tuples
    # each match = (sql_fragment, ending_char)
    sql_statements = [m.strip() for m in matches]
    return sql_statements


def verify_queries_exist(text):
    """
    Main function:
    - tries JSON parsing first
    - falls back to regex extraction
    - returns:
        {
            "method": "json" | "regex" | "none",
            "sql_queries": [...],
            "count": int
        }
    """

    data, ok_json = try_parse_json(text)

    if ok_json and isinstance(data, list):
        # Extract SQL fields
        sqls = []
        for item in data:
            if isinstance(item, dict) and "sql" in item:
                val = item["sql"]
                if isinstance(val, str) and val.strip().upper().startswith("SELECT"):
                    sqls.append(val.strip())

        if sqls:
            return {
                "method": "json",
                "sql_queries": sqls,
                "count": len(sqls)
            }

    sqls = extract_sql_with_regex(text)
    if sqls:
        return {
            "method": "regex",
            "sql_queries": sqls,
            "count": len(sqls)
        }
    return {
        "method": "none",
        "sql_queries": [],
        "count": 0
    }

def print_verification_report(result):
    print(f"Detection method: {result['method']}")
    print(f"Found SQL queries: {result['count']}")
    print("-" * 40)

    for i, q in enumerate(result["sql_queries"], start=1):
        print(f"[Query {i}]\n{q}\n")
