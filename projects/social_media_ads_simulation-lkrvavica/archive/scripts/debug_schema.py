import duckdb
con = duckdb.connect("data/databases/simulation_db.duckdb")
try:
    print(con.execute("PRAGMA table_info('interactions')").df())
except Exception as e:
    print(f"Error: {e}")
con.close()
