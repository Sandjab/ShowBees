import sqlite3
from contextlib import closing


def create(db_path):
    with closing(sqlite3.connect(db_path)) as db:
        c = db.cursor()

        c.executescript(
            """
            -- Create table to store AudioDataset configuration
            CREATE TABLE IF NOT EXISTS config(
                sample_rate INT NOT NULL,
                duration    REAL NOT NULL,
                overlap     REAL NOT NULL,
                nb_samples  INT NOT NULL
            );

            -- Create table to store AudioDataset list of source files names
            CREATE TABLE IF NOT EXISTS filenames(name TEXT);

            -- Create table to store AudioDataset samples information
            CREATE TABLE IF NOT EXISTS samples(
                parent_id   INT DEFAULT 0,
                name        TEXT,
                file_id     INT,
                augment_id  INT,
                start_t     REAL,
                end_t       REAL
            );

            -- Create table to store augmentation information
            CREATE TABLE IF NOT EXISTS augmentations(
                method          TEXT NOT NULL,
                int_param1      INT,
                int_param2      INT,
                float_param1    REAL,
                float_param2    REAL,
                next            INT
            );

            -- Create table to store samples attributes
            CREATE TABLE IF NOT EXISTS attributes(sample_id INT PRIMARY KEY);

            -- Create table to store samples labels
            CREATE TABLE IF NOT EXISTS labels(sample_id INT PRIMARY KEY);

            -- Create table to store samples features
            CREATE TABLE IF NOT EXISTS features(sample_id INT PRIMARY KEY);
            """
        )

        db.commit()

    db.close()
    return


def list_columns(db, table_name):
    c = db.cursor()
    c.execute(F"SELECT name, type, pk FROM PRAGMA_TABLE_INFO('{table_name}');")
    results = c.fetchall()
    c.close()
    return results


def list_column_names(db, table_name, but=()):
    rows = list_columns(db, table_name)
    results = []
    for row in rows:
        name, _, _ = row
        if name not in but:
            results.append(name)
    return results


def column_exists(db, table_name, column_name):
    c = db.cursor()
    c.execute(
        F"""
        SELECT 1 FROM PRAGMA_TABLE_INFO('{table_name}')
        WHERE name='{column_name}';
        """
    )
    result = (c.fetchone() is not None)
    c.close()
    return result


def add_column(db, table_name, column_name, column_type):
    if not column_exists(db, table_name, column_name):
        c = db.cursor()
        c.execute(
            F"""ALTER TABLE {table_name}
            ADD COLUMN {column_name} {column_type} """
        )
        c.close()
        db.commit()
        return True
    return False


def del_column(db, table_name, column_name):
    if (column_exists(db, table_name, column_name)):
        rows = list_columns(db, table_name)
        print("rows = ", rows)
        cols = []
        typed_cols = []
        for row in rows:
            name, type, pk = row
            if (name != column_name):
                cols.append(name)
                tc = name + " " + type + (" PRIMARY KEY" if pk else "")
                typed_cols.append(tc)

        str_cols = ",".join(cols)
        str_typed_cols = ",".join(typed_cols)

        c = db.cursor()
        script = F"""
            PRAGMA foreign_keys=off;
            BEGIN TRANSACTION;
            CREATE TABLE IF NOT EXISTS temp_{table_name} ({str_typed_cols});
            INSERT INTO temp_{table_name}({str_cols})
                SELECT {str_cols} FROM {table_name};
            DROP TABLE {table_name};
            ALTER TABLE temp_{table_name} RENAME TO {table_name};
            COMMIT;
            PRAGMA foreign_keys=on;
            """
        c.executescript(script)
        c.close()
        db.commit()
