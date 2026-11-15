"""Full collation safety matrix validated against SQLite semantics."""
import sqlite3, sys, collections
sys.path.insert(0, "/workspace")
import sqlglot
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.eliminate_joins import eliminate_joins

schema = {"l": {"k": "TEXT"}, "r": {"k": "TEXT"}}
con = sqlite3.connect(":memory:")
con.executescript("CREATE TABLE l(k TEXT); INSERT INTO l VALUES ('a'),('A'),('a ');"
                   "CREATE TABLE r(k TEXT); INSERT INTO r VALUES ('a'),('A'),('a '),('b')")

collations = ["BINARY", "NOCASE", "RTRIM"]
results = []
for unique_coll in collations:
    for join_coll in collations + [None]:
        right = f"SELECT DISTINCT r.k COLLATE {unique_coll} AS k FROM r"
        on = f"l.k COLLATE {join_coll} = q.k" if join_coll else "l.k = q.k"
        sql = f"SELECT l.k FROM l LEFT JOIN ({right}) q ON {on}"
        e = annotate_types(qualify(sqlglot.parse_one(sql, dialect="sqlite"), schema=schema, dialect="sqlite"),
                           schema=schema, dialect="sqlite")
        out = eliminate_joins(e.copy(), dialect="sqlite", schema=schema).sql("sqlite")
        eliminated = "JOIN" not in out.upper()
        # engine ground truth
        orig = collections.Counter(tuple(x) for x in con.execute(sql).fetchall())
        elim_sql = "SELECT l.k FROM l"
        got = collections.Counter(tuple(x) for x in con.execute(elim_sql).fetchall())
        engine_safe = orig == got
        verdict = "OK" if eliminated == (engine_safe) else "MISMATCH"
        results.append((verdict, unique_coll, join_coll or "DEFAULT(BIN)", eliminated, engine_safe))
        if verdict != "OK":
            print("MISMATCH", unique_coll, join_coll, eliminated, engine_safe, orig, got)

for row in results:
    print(f"{row[0]:8s} unique={row[1]:7s} join={row[2]:12s} eliminated={row[3]!s:5s} engine_safe={row[4]}")
print("all", len(results), "matrix cells correct:", all(r[0] == "OK" for r in results))
