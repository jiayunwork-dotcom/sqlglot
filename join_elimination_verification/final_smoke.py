import sys, collections
sys.path.insert(0, "/workspace")
import duckdb, sqlite3
from sqlglot.optimizer import optimize

con = duckdb.connect()
con.execute("CREATE TABLE l(k INTEGER); INSERT INTO l VALUES (1),(2)")
con.execute("CREATE TABLE r(k VARCHAR); INSERT INTO r VALUES ('1'),('01'),('2')")
sql = "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k=q.k ORDER BY l.k"
opt = optimize(sql, schema={"l": {"k": "INTEGER"}, "r": {"k": "VARCHAR"}}, dialect="duckdb").sql("duckdb")
assert con.execute(sql).fetchall() == con.execute(opt).fetchall() == [(1,), (1,), (2,)]
assert "LEFT JOIN" in opt.upper()
print("case 1 OK")

s = sqlite3.connect(":memory:")
s.executescript("CREATE TABLE l(k TEXT); INSERT INTO l VALUES ('a');"
                "CREATE TABLE r(k TEXT); INSERT INTO r VALUES ('a'),('A')")
for sql in (
    "SELECT l.k FROM l LEFT JOIN (SELECT DISTINCT r.k FROM r) q ON l.k COLLATE NOCASE=q.k",
    "SELECT l.k FROM l LEFT JOIN (SELECT r.k FROM r GROUP BY r.k) q ON l.k COLLATE NOCASE=q.k",
    "WITH q AS (SELECT DISTINCT r.k FROM r) SELECT l.k FROM l LEFT JOIN q ON l.k COLLATE NOCASE=q.k",
):
    opt = optimize(sql, schema={"l": {"k": "TEXT"}, "r": {"k": "TEXT"}}, dialect="sqlite").sql("sqlite")
    assert collections.Counter(map(tuple, s.execute(sql).fetchall())) == collections.Counter(map(tuple, s.execute(opt).fetchall())) == collections.Counter({("a",): 2})
    assert "LEFT JOIN" in opt.upper()
print("case 2 OK: all three shapes preserve both rows")
