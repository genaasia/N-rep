import sqlglot


def extract_table_names(sql):
    parsed = sqlglot.parse_one(sql)
    tables = parsed.find_all(sqlglot.expressions.Table)
    return {table.name for table in tables}


def extract_column_names(sql):
    parsed = sqlglot.parse_one(sql)
    columns = parsed.find_all(sqlglot.expressions.Column)

    column_info = set()
    for col in columns:
        if col.table:
            column_info.add(f"{col.table}.{col.name}")
        else:
            column_info.add(col.name)

    return column_info


def main():
    sql = """WITH time_in_seconds AS ( SELECT T1.positionOrder, CASE WHEN T1.positionOrder = 1 THEN (CAST(SUBSTR(T1.time, 1, 1) AS REAL) * 3600) + (CAST(SUBSTR(T1.time, 3, 2) AS REAL) * 60) + CAST(SUBSTR(T1.time, 6) AS REAL) ELSE CAST(SUBSTR(T1.time, 2) AS REAL) END AS time_seconds FROM results AS T1 INNER JOIN races AS T2 ON T1.raceId = T2.raceId WHERE T2.name = 'Australian Grand Prix' AND T1.time IS NOT NULL AND T2.year = 2008 ), champion_time AS ( SELECT time_seconds FROM time_in_seconds WHERE positionOrder = 1), last_driver_incremental AS ( SELECT time_seconds FROM time_in_seconds WHERE positionOrder = (SELECT MAX(positionOrder) FROM time_in_seconds) ) SELECT (CAST((SELECT time_seconds FROM last_driver_incremental) AS REAL) * 100) / (SELECT time_seconds + (SELECT time_seconds FROM last_driver_incremental) FROM champion_time)"""
    tables = extract_table_names(sql)
    print(tables)
    cols = extract_column_names(sql)
    print(cols)


if __name__ == "__main__":
    main()
