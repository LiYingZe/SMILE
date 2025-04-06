import psycopg2
import time
import csv
import argparse

# 数据库连接信息
DB_NAME = "sql_data_test"
DB_USER = "postgres"
DB_PASSWORD = "123456"
DB_HOST = "localhost"
DB_PORT = "5432"

INDEX_TYPES = ["prefix", "suffix", "hash", "trgm", "gist"]


def get_text_columns(cur, table):
    cur.execute(f"""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = %s 
        AND data_type IN ('text', 'character varying')
        AND column_name <> 'id';
    """, (table,))
    return [row[0] for row in cur.fetchall()]


def create_index(cur, conn, table_index, col, index_type):
    start_time = time.time()
    if index_type == "prefix":
        cur.execute(f"""
            CREATE INDEX {table_index}_{col}_prefix_idx 
            ON {table_index} (LEFT({col}, 5));
        """)
    elif index_type == "suffix":
        cur.execute(f"""
            ALTER TABLE {table_index} ADD COLUMN {col}_rev TEXT;
            UPDATE {table_index} SET {col}_rev = REVERSE({col});
            CREATE INDEX {table_index}_{col}_suffix_idx 
            ON {table_index} ({col}_rev);
        """)
    elif index_type == "hash":
        cur.execute(f"""
            CREATE INDEX {table_index}_{col}_hash_idx 
            ON {table_index} USING hash ({col});
        """)
    elif index_type == "trgm":
        cur.execute(f"""
            CREATE INDEX {table_index}_{col}_trgm_idx 
            ON {table_index} USING gin ({col} gin_trgm_ops);
        """)
    elif index_type == "gist":
        cur.execute(f"""
            CREATE INDEX {table_index}_{col}_gist_idx 
            ON {table_index} USING GIST ({col} gist_trgm_ops);
        """)
    conn.commit()
    end_time = time.time()
    execution_time = round((end_time - start_time) * 1000, 2)
    index_name = f"{table_index}_{col}_{index_type}_idx"
    print(f"Index {index_name} created in {execution_time} ms.")
    return [index_name, execution_time]


def main(csv_filename):
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()

    cur.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    """)
    tables = [row[0] for row in cur.fetchall() if not row[0].endswith('_id_seq')]

    index_creation_times = []

    for table in tables:
        text_columns = get_text_columns(cur, table)

        for index_type in INDEX_TYPES:
            table_index = f"{table}_{index_type}"
            print(f"\nProcessing table: {table_index}")

            cur.execute(f"DROP TABLE IF EXISTS {table_index};")
            cur.execute(f"CREATE TABLE {table_index} AS TABLE {table};")
            conn.commit()

            for col in text_columns:
                index_info = create_index(cur, conn, table_index, col, index_type)
                index_creation_times.append(index_info)

    cur.close()
    conn.close()

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["index_name", "time(ms)"])
        writer.writerows(index_creation_times)

    print(f"\nAll indexed tables created successfully! Index creation times saved in {csv_filename}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and time index creation on PostgreSQL tables.")
    parser.add_argument("--csv_filename", type=str, required=True, help="Output CSV filename to store index creation times.")
    args = parser.parse_args()

    main(args.csv_filename)
