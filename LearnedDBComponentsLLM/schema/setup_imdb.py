import os
import gzip
import requests
import shutil
from pathlib import Path
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Load env vars
load_dotenv("../.env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1111")
DB_NAME = "imdb" # Target DB

# URLs
BASE_URL = "https://datasets.imdbws.com"
FILES = [
    "name.basics.tsv.gz",
    "title.akas.tsv.gz",
    "title.basics.tsv.gz",
    "title.crew.tsv.gz",
    "title.episode.tsv.gz",
    "title.principals.tsv.gz",
    "title.ratings.tsv.gz"
]

DATA_DIR = Path("imdb-datasets")

def download_file(url, dest_path):
    if dest_path.exists():
        print(f"File {dest_path} already exists. Skipping download.")
        return
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def decompress_file(gz_path, dest_path):
    if dest_path.exists():
        print(f"File {dest_path} already exists. Skipping decompression.")
        return
    print(f"Decompressing {gz_path}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def get_conn(db_name="postgres"):
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=db_name
    )
    conn.autocommit = True
    return conn

def setup_database():
    print("Connecting to postgres...")
    try:
        conn = get_conn("postgres")
        cur = conn.cursor()
        
        # Terminate existing connections
        print(f"Terminating existing connections to {DB_NAME}...")
        cur.execute(sql.SQL("""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = {} AND pid <> pg_backend_pid();
        """).format(sql.Literal(DB_NAME)))
        
        # Drop and Create DB
        print(f"Recreating database {DB_NAME}...")
        cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(DB_NAME)))
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")
        exit(1)

def create_tables_and_load_data():
    print(f"Connecting to {DB_NAME}...")
    try:
        conn = get_conn(DB_NAME)
        cur = conn.cursor()
        
        # Create Tables
        tables_sql = [
            "CREATE TABLE title_ratings (tconst VARCHAR(10),average_rating NUMERIC,num_votes integer);",
            "CREATE TABLE name_basics (nconst varchar(10), primaryName text, birthYear smallint, deathYear smallint, primaryProfession text, knownForTitles text );",
            "CREATE TABLE title_akas (titleId TEXT, ordering INTEGER, title TEXT, region TEXT, language TEXT, types TEXT, attributes TEXT, isOriginalTitle BOOLEAN);",
            "CREATE TABLE title_basics (tconst TEXT, titleType TEXT, primaryTitle TEXT, originalTitle TEXT, isAdult BOOLEAN, startYear SMALLINT, endYear SMALLINT, runtimeMinutes INTEGER, genres TEXT);",
            "CREATE TABLE title_crew (tconst TEXT, directors TEXT, writers TEXT);",
            "CREATE TABLE title_episode (const TEXT, parentTconst TEXT, seasonNumber TEXT, episodeNumber TEXT);",
            "CREATE TABLE title_principals (tconst TEXT, ordering INTEGER, nconst TEXT, category TEXT, job TEXT, characters TEXT);"
        ]
        
        for stmt in tables_sql:
            print(f"Executing: {stmt}")
            cur.execute(stmt)
            
        # Load Data
        # Map filenames to table names
        file_map = {
            "title.ratings.tsv": "title_ratings",
            "name.basics.tsv": "name_basics",
            "title.akas.tsv": "title_akas",
            "title.basics.tsv": "title_basics",
            "title.crew.tsv": "title_crew",
            "title.episode.tsv": "title_episode",
            "title.principals.tsv": "title_principals"
        }
        
        for filename, table in file_map.items():
            file_path = DATA_DIR / filename
            print(f"Loading {table} from {file_path}...")
            
            with open(file_path, "r", encoding="utf-8") as f:
                # Use COPY ... FROM STDIN with csv options
                # The files are TSV, NULL is \N
                copy_sql = f"COPY {table} FROM STDIN WITH (FORMAT CSV, DELIMITER E'\t', QUOTE E'\b', NULL '\\N', HEADER)"
                try:
                    cur.copy_expert(copy_sql, f)
                    print(f"Loaded {table} successfully.")
                except Exception as e:
                    print(f"Error loading {table}: {e}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error in table creation or data loading: {e}")
        exit(1)

def main():
    DATA_DIR.mkdir(exist_ok=True)
    
    # 1. Download
    for fname in FILES:
        download_file(BASE_URL + "/" + fname, DATA_DIR / fname)
        
    # 2. Decompress
    for fname in FILES:
        base_name = fname.replace(".gz", "")
        decompress_file(DATA_DIR / fname, DATA_DIR / base_name)
        
    # 3. DB Setup
    setup_database()
    
    # 4. Tables & Data
    create_tables_and_load_data()
    
    print("Done!")

if __name__ == "__main__":
    main()
