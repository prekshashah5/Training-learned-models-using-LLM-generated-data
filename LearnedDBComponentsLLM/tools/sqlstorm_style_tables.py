import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running as a standalone script from repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.query_generator import SchemaValidator


LEVELS = {"low": 1, "medium": 2, "high": 3}
LEVEL_NAMES = {1: "low", 2: "medium", 3: "high"}


def _split_top_level_csv(expr: str) -> List[str]:
    parts = []
    cur = []
    depth = 0
    in_quote = False
    quote_char = ""

    for ch in expr:
        if ch in ("'", '"'):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
            cur.append(ch)
            continue

        if in_quote:
            cur.append(ch)
            continue

        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == "," and depth == 0:
            token = "".join(cur).strip()
            if token:
                parts.append(token)
            cur = []
        else:
            cur.append(ch)

    token = "".join(cur).strip()
    if token:
        parts.append(token)
    return parts


def _extract_from_clause(sql: str) -> str:
    m = re.search(r"\bfrom\b\s+(.+?)(?:\bwhere\b|\bgroup\b\s+by\b|\border\b\s+by\b|\blimit\b|$)", sql, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_select_clause(sql: str) -> str:
    m = re.search(r"\bselect\b\s+(.+?)\s+\bfrom\b", sql, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_alias_map(sql: str) -> Dict[str, str]:
    from_clause = _extract_from_clause(sql)
    if not from_clause:
        return {}

    parts = re.split(r"\b(?:inner\s+|left\s+|right\s+|full\s+|cross\s+)?join\b|,", from_clause, flags=re.IGNORECASE)
    alias_to_table: Dict[str, str] = {}

    for part in parts:
        part = re.split(r"\bon\b", part, flags=re.IGNORECASE)[0].strip()
        if not part:
            continue
        toks = part.split()
        if not toks:
            continue

        table = toks[0].strip().lower()
        alias = table
        if len(toks) >= 2:
            if toks[1].lower() == "as" and len(toks) >= 3:
                alias = toks[2].strip().lower()
            elif toks[1].lower() not in {"on", "where", "join"}:
                alias = toks[1].strip().lower()

        alias_to_table[alias] = table

    return alias_to_table


def _count_groupby_items(sql: str) -> int:
    m = re.search(r"\bgroup\s+by\b\s+(.+?)(?:\border\s+by\b|\blimit\b|$)", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return 0
    items = _split_top_level_csv(m.group(1).strip())
    return len(items)


def _count_orderby_items(sql: str) -> int:
    m = re.search(r"\border\s+by\b\s+(.+?)(?:\blimit\b|$)", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return 0
    items = _split_top_level_csv(m.group(1).strip())
    return len(items)


def _count_join_conditions(sql: str) -> int:
    return len(re.findall(r"\b\w+\.\w+\s*=\s*\w+\.\w+\b", sql, re.IGNORECASE))


def _classify_types(sql: str, schema_validator: SchemaValidator) -> Tuple[int, int, int]:
    alias_map = _extract_alias_map(sql)
    refs = re.findall(r"\b([a-z_][\w]*)\.([a-z_][\w]*)\b", sql, re.IGNORECASE)

    low = 0
    med = 0
    high = 0

    for alias, col in refs:
        table = alias_map.get(alias.lower())
        if not table:
            continue
        info = schema_validator.tables.get(table.lower())
        if not info:
            continue
        col_type = info.get("columns", {}).get(col.lower())
        if not col_type:
            continue

        ct = col_type.lower()
        if ct in {"integer", "int", "smallint", "bigint", "serial", "bigserial", "numeric", "decimal", "real", "float", "double", "double precision", "boolean", "bool", "text", "varchar", "char", "character varying", "character", "date", "timestamp"}:
            low += 1
        elif "[]" in ct or "array" in ct:
            med += 1
        elif ct in {"record", "json", "jsonb", "timestamptz", "timestamp with time zone"}:
            high += 1
        else:
            low += 1

    return low, med, high


def extract_features(sql: str, schema_validator: SchemaValidator) -> Dict[str, float]:
    s = sql.strip()
    s_lower = s.lower()

    select_clause = _extract_select_clause(s)
    select_count = len(_split_top_level_csv(select_clause)) if select_clause else 0

    from_clause = _extract_from_clause(s)
    table_scan = len([p for p in re.split(r"\b(?:inner\s+|left\s+|right\s+|full\s+|cross\s+)?join\b|,", from_clause, flags=re.IGNORECASE) if p.strip()]) if from_clause else 0

    join_count = _count_join_conditions(s)
    groupby_count = _count_groupby_items(s)
    sort_count = _count_orderby_items(s)

    window_count = len(re.findall(r"\bover\s*\(", s_lower))
    setop_count = len(re.findall(r"\bunion\b|\bintersect\b|\bexcept\b", s_lower))
    array_unnest_count = len(re.findall(r"\bunnest\s*\(", s_lower))
    iteration_count = len(re.findall(r"\bwith\s+recursive\b|\biterate\b|\bconnect\s+by\b", s_lower))
    regexsplit_count = len(re.findall(r"\bregexp_split\w*\s*\(", s_lower))
    map_count = len(re.findall(r"\bmap\s*\(", s_lower))

    comparison_count = len(re.findall(r"(?<![<>=!])(?:<=|>=|<>|!=|=|<|>)(?![<>=])", s))
    cast_count = len(re.findall(r"\bcast\s*\(", s_lower)) + s.count("::")
    arith_simple_count = len(re.findall(r"(?<=[\w\)])\s*[+\-*/]\s*(?=[\w\(])", s))

    nulls_count = len(re.findall(r"\bis\s+null\b|\bis\s+not\s+null\b|\bcoalesce\s*\(|\bnullif\s*\(", s_lower))
    strings_count = len(re.findall(r"'([^']|'')*'", s))
    date_count = len(re.findall(r"\bdate\b|\btimestamp\b|\binterval\b|\bdate_trunc\b|\bextract\s*\(|\bto_date\s*\(", s_lower))
    array_count = len(re.findall(r"\barray\s*\[|\bunnest\s*\(|::\w+\[\]", s_lower))
    arith_complex_count = len(re.findall(r"\bpower\s*\(|\bsqrt\s*\(|\bmod\s*\(|\bexp\s*\(|\bln\s*\(", s_lower))
    regex_count = len(re.findall(r"\bregexp\w*\b|\bregex\b|\bsimilar\s+to\b|\brlike\b", s_lower))
    json_count = len(re.findall(r"->>|->|#>>|#>|\bjson\w*\b|\bjsonb\w*\b", s_lower))

    type_low, type_medium, type_high = _classify_types(s, schema_validator)

    join_inner = join_count + len(re.findall(r"\binner\s+join\b", s_lower))
    join_outer = len(re.findall(r"\bleft\s+join\b|\bright\s+join\b|\bfull\s+join\b|\bouter\s+join\b", s_lower))
    join_semi = len(re.findall(r"\bsemi\s+join\b", s_lower))
    join_anti = len(re.findall(r"\banti\s+join\b", s_lower))
    join_single = len(re.findall(r"\bsingle\s+join\b", s_lower))
    join_mark = len(re.findall(r"\bmark\s+join\b", s_lower))

    operator_level = LEVELS["low"]
    if iteration_count > 0 or regexsplit_count > 0 or array_unnest_count > 0:
        operator_level = LEVELS["high"]
    elif window_count > 0 or setop_count > 0:
        operator_level = LEVELS["medium"]

    expr_level = LEVELS["low"]
    if regex_count > 0 or json_count > 0:
        expr_level = LEVELS["high"]
    elif nulls_count > 0 or strings_count > 0 or date_count > 0 or array_count > 0 or arith_complex_count > 0:
        expr_level = LEVELS["medium"]

    type_level = LEVELS["low"]
    if type_high > 0:
        type_level = LEVELS["high"]
    elif type_medium > 0:
        type_level = LEVELS["medium"]

    join_type_level = LEVELS["low"]
    if join_mark > 0:
        join_type_level = LEVELS["high"]
    elif join_semi > 0 or join_anti > 0 or join_single > 0:
        join_type_level = LEVELS["medium"]

    joins_level = LEVELS["low"] if join_count <= 3 else (LEVELS["medium"] if join_count <= 8 else LEVELS["high"])
    groupby_level = LEVELS["low"] if groupby_count <= 1 else (LEVELS["medium"] if groupby_count <= 3 else LEVELS["high"])

    overall_level = max(operator_level, expr_level, type_level, join_type_level, joins_level, groupby_level)
    complexity = LEVEL_NAMES[overall_level]

    return {
        "query_text_len": float(len(s)),
        "operators": float(comparison_count),
        "tablescan": float(table_scan),
        "join": float(join_count),
        "sort": float(sort_count),
        "groupby": float(groupby_count),
        "select": float(select_count),
        "window": float(window_count),
        "setoperation": float(setop_count),
        "arrayunnest": float(array_unnest_count),
        "iteration": float(iteration_count),
        "regexsplit": float(regexsplit_count),
        "expr_total": float(comparison_count + cast_count + arith_simple_count + nulls_count + strings_count + date_count + array_count + arith_complex_count + regex_count + json_count),
        "expr_comparison": float(comparison_count),
        "expr_cast": float(cast_count),
        "expr_arithmetic_simple": float(arith_simple_count),
        "expr_nulls": float(nulls_count),
        "expr_strings": float(strings_count),
        "expr_date": float(date_count),
        "expr_array": float(array_count),
        "expr_arithmetic_complex": float(arith_complex_count),
        "expr_regex": float(regex_count),
        "expr_json": float(json_count),
        "types_low": float(type_low),
        "types_medium": float(type_medium),
        "types_high": float(type_high),
        "join_inner": float(join_inner),
        "join_outer": float(join_outer),
        "join_semi": float(join_semi),
        "join_anti": float(join_anti),
        "join_single": float(join_single),
        "join_mark": float(join_mark),
        "complexity": complexity,
    }


def _mean(rows: List[Dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(r.get(key, 0.0)) for r in rows) / len(rows), 3)


def build_table6(rows: List[Dict[str, float]]) -> List[Dict[str, object]]:
    buckets = {
        "low": [r for r in rows if r["complexity"] == "low"],
        "medium": [r for r in rows if r["complexity"] == "medium"],
        "high": [r for r in rows if r["complexity"] == "high"],
        "total": rows,
    }

    metrics = [
        ("Query Text", "query_text_len"),
        ("Operators", "operators"),
        ("TableScan", "tablescan"),
        ("Join", "join"),
        ("Sort", "sort"),
        ("GroupBy", "groupby"),
        ("Select", "select"),
        ("Window", "window"),
        ("SetOperation", "setoperation"),
        ("ArrayUnnest", "arrayunnest"),
        ("Iteration", "iteration"),
        ("RegexSplit", "regexsplit"),
        ("Expressions", "expr_total"),
        ("comparison", "expr_comparison"),
        ("cast", "expr_cast"),
        ("arithmetic_simple", "expr_arithmetic_simple"),
        ("nulls", "expr_nulls"),
        ("strings", "expr_strings"),
        ("date", "expr_date"),
        ("array", "expr_array"),
        ("arithmetic_complex", "expr_arithmetic_complex"),
        ("regex", "expr_regex"),
        ("json", "expr_json"),
    ]

    out = []
    for label, key in metrics:
        out.append(
            {
                "metric": label,
                "low": _mean(buckets["low"], key),
                "medium": _mean(buckets["medium"], key),
                "high": _mean(buckets["high"], key),
                "total": _mean(buckets["total"], key),
            }
        )
    return out


def build_table5_summary(rows: List[Dict[str, float]]) -> Dict[str, object]:
    total = len(rows)
    c_low = sum(1 for r in rows if r["complexity"] == "low")
    c_med = sum(1 for r in rows if r["complexity"] == "medium")
    c_high = sum(1 for r in rows if r["complexity"] == "high")

    return {
        "total_queries": total,
        "complexity_distribution": {
            "low": c_low,
            "medium": c_med,
            "high": c_high,
        },
        "thresholds": {
            "joins": {"low": "<=3", "medium": "<=8", "high": ">8"},
            "groupby": {"low": "<=1", "medium": "<=3", "high": ">3"},
        },
    }


def _print_markdown_table(table6: List[Dict[str, object]]) -> None:
    print("\nTable (SQLStorm-style): Average number of operators, expressions, and query text size\n")
    print("| Metric | low | med | high | Total |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for r in table6:
        print(f"| {r['metric']} | {r['low']:.3f} | {r['medium']:.3f} | {r['high']:.3f} | {r['total']:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLStorm-style comparison tables from generated queries")
    parser.add_argument("--input", type=str, default="", help="Path to queries_*.json file; default is latest in generated_queries/")
    parser.add_argument("--schema", type=str, default="schema/IMDB_schema.txt", help="Schema DDL path for type classification")
    parser.add_argument("--out-dir", type=str, default="generated_queries", help="Output folder")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        query_path = Path(args.input)
    else:
        candidates = sorted(out_dir.glob("queries_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError("No queries_*.json found in generated_queries/")
        query_path = candidates[-1]

    with query_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    queries = [q for q in raw if isinstance(q, str)]
    if not queries:
        raise RuntimeError("Input file does not contain SQL string queries")

    schema_text = Path(args.schema).read_text(encoding="utf-8")
    schema_validator = SchemaValidator(schema_text)

    rows: List[Dict[str, float]] = []
    for idx, sql in enumerate(queries):
        feats = extract_features(sql, schema_validator)
        record: Dict[str, object] = {"index": idx, "sql": sql}
        record.update(feats)
        rows.append(record)  # type: ignore[arg-type]

    table6 = build_table6(rows)  # type: ignore[arg-type]
    table5 = build_table5_summary(rows)  # type: ignore[arg-type]

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = query_path.stem

    raw_csv = out_dir / f"sqlstorm_raw_{stem}_{ts}.csv"
    table6_csv = out_dir / f"sqlstorm_table6_{stem}_{ts}.csv"
    table5_json = out_dir / f"sqlstorm_table5_{stem}_{ts}.json"

    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with table6_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric", "low", "medium", "high", "total"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table6)

    with table5_json.open("w", encoding="utf-8") as f:
        json.dump(table5, f, indent=2)

    print(f"Analyzed file: {query_path}")
    print(f"Total queries: {len(rows)}")
    print(f"Saved raw per-query data: {raw_csv}")
    print(f"Saved table6-style aggregates: {table6_csv}")
    print(f"Saved table5-style summary: {table5_json}")

    _print_markdown_table(table6)


if __name__ == "__main__":
    main()
