import psycopg2
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import json
import os
import re
from tqdm import tqdm

global host, dbname, user, password, port
workload_dir = "./queries"
workloads = [f for f in os.listdir(workload_dir) if f.endswith(".txt")]
print(workloads)

def get_column_name(table_name, connection_params):
    """
    识别非 ID 的列名
    """
    try:
        connection = psycopg2.connect(**connection_params)
        cursor = connection.cursor()
        
        query = f"""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = '{table_name}' AND column_name NOT IN ('id', 'ID');
        """
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if result:
            return result[0][0]
        else:
            raise ValueError(f"No valid column found for table {table_name}")
    except Exception as e:
        print(f"Error identifying column name for table {table_name}: {e}")
        return None

def execute_query(pattern, table_name, column_name, connection_params):
    try:
        connection = psycopg2.connect(**connection_params)
        cursor = connection.cursor()
        
        if "_gist_new" in table_name or "_gin_new" in table_name:
            if "_gist_new" in table_name:
                table_name = table_name.replace("_gist_new", "_gist")
            elif "_gin_new" in table_name:
                table_name = table_name.replace("_gin_new", "_trgm")
            query = f"""
            SELECT {column_name} FROM {table_name} 
            WHERE {column_name} % '{pattern}' 
            LIMIT 128;
            """
            start_time = time.time()
            cursor.execute(query)
        elif "_gist" in table_name:
            query = f"""
            SELECT {column_name} FROM {table_name} 
            WHERE {column_name} LIKE %s 
            LIMIT 128;
            """
            start_time = time.time()
            cursor.execute(query, (f"{pattern}",))
        elif "_suffix" in table_name:
            print(pattern)
            print('\n')
            pattern = pattern[::-1]
            print(pattern)
            query = f"""
            SELECT {column_name} FROM {table_name} 
            WHERE {column_name}_rev LIKE %s;
            """
            start_time = time.time()
            cursor.execute(query, (f"{pattern}",))
        else:
            query = f"""
            SELECT {column_name} FROM {table_name} 
            WHERE {column_name} LIKE %s;
            """
            start_time = time.time()
            cursor.execute(query, (f"{pattern}",))

        result = cursor.fetchall()
        end_time = time.time()
        
        results = [row[0] for row in result]
        cursor.close()
        connection.close()
        
        return pattern, results, start_time, end_time
    except Exception as e:
        print(f"Error executing query with pattern {pattern}: {e}")
        return pattern, [], 0, 0

def main():
    
    concurrency_level = 6
    
    for workload in workloads:
        workload_path = os.path.join(workload_dir, workload)
        with open(workload_path, 'r', encoding='utf-8') as f:
            like_patterns = [line.strip() for line in f.readlines() if line.strip()]

        match = re.search(r'^(.*?)_ipct', workload)
        table_name_prefix = match.group(1).lower() if match else workload.split('.')[0].lower()
        print(f"Processing table: {table_name_prefix}")
        base_table_name = table_name_prefix
        print(base_table_name)
        
        if base_table_name == 'emails':
            host = 'localhost'
            dbname = 'emails'
            user = 'postgres'
            password = '123456'
            port = 5432
            connection_params = {
            'host': host,
            'dbname': dbname,
            'user': user,
            'password': password,
            'port': port
        }
        else:
            host = 'localhost'
            dbname = 'sql_data_test'
            user = 'postgres'
            password = '123456'
            port = 5432
            connection_params = {
            'host': host,
            'dbname': dbname,
            'user': user,
            'password': password,
            'port': port
        }
        # extensions = ["","_prefix","_suffix","_hash","_gist_new","_gin_new","_gist", "_trgm"]
        extensions = ["","_prefix","_suffix","_hash","_gist_new","_gin_new","_gist", "_trgm"]
        for ext in extensions:
            table_name = f"{base_table_name}{ext}"
            if "_gist_new" in table_name:
                re_table_name = table_name.replace("_gist_new", "_gist")
                print(re_table_name)
                column_name = get_column_name(re_table_name, connection_params)
            elif "_gin_new" in table_name:
                re_table_name = table_name.replace("_gin_new", "_trgm")
                column_name = get_column_name(re_table_name, connection_params)
            else:
                column_name = get_column_name(table_name, connection_params)
            
            if not column_name:
                print(f"Skipping {table_name} due to missing column name.")
                continue
            
            os.makedirs(table_name, exist_ok=True)
            
            all_results = {}
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=concurrency_level) as executor:
                future_to_pattern = {executor.submit(execute_query, pattern, table_name, column_name, connection_params): pattern for pattern in like_patterns}
                
                for future in tqdm(future_to_pattern, desc=f"Processing {table_name}", total=len(like_patterns)):
                    pattern, results, query_start, query_end = future.result()
                    if len(results) > 999:
                        results = []
                        results.append('out 1000')
                    all_results[pattern] = {
                        "results": results,
                        "start_time": query_start,
                        "end_time": query_end,
                        "duration_ms": round((query_end - query_start) * 1000, 2)
                    }
            
            output_file = os.path.join(table_name, workload.replace(".txt", ".json"))
            with open(output_file, "w") as json_file:
                json.dump(all_results, json_file, ensure_ascii=False, indent=4)
            
            print(f"Results saved to {output_file}")
            end_time = time.time()
            print(f"Time taken: {round((end_time - start_time) / 60, 2)} minutes")

if __name__ == "__main__":
    main()
