import sqlite3
import pickle
import numpy as np
import re
from contextlib import closing
# import time

RESERVED_COL_NAMES = ('rowid', 'name', 'file_id', 'start_t', 'end_t')

sqlite3.register_adapter(list, pickle.dumps)
sqlite3.register_adapter(set, pickle.dumps)
sqlite3.register_adapter(dict, pickle.dumps)
sqlite3.register_adapter(np.ndarray, pickle.dumps)
sqlite3.register_converter("feature", pickle.loads)


def create(db_path):
    with closing(sqlite3.connect(db_path)) as db:
        c = db.cursor()

        c.executescript(
            """
            -- Create table to store AudioDataset configuration
            CREATE TABLE IF NOT EXISTS config(
                ds_name     TEXT NOT NULL,
                sample_rate INT NOT NULL,
                duration    REAL NOT NULL,
                overlap     REAL NOT NULL,
                nb_samples  INT NOT NULL
            );

            -- Create table to store AudioDataset samples information
            CREATE TABLE IF NOT EXISTS samples(
                name        TEXT,
                file_id     INT,
                start_t     REAL,
                end_t       REAL
            );

            -- Create table to store additional fields name
            CREATE TABLE IF NOT EXISTS dictionary(
                name    TEXT PRIMARY KEY,
                type    TEXT
            );

            -- Create table to store AudioDataset list of source files names
            CREATE TABLE IF NOT EXISTS filenames(name TEXT);
            """
        )
        db.commit()

    db.close()
    return


def assert_valid_name(name):
    assert re.match("^[a-z0-9_]+$", name), \
        F"Invalid name ({name}. Fields names " \
        "must contain only lovercase ascii letters (a-z), " \
        "digits (0-9) and underscore (_))"
    assert name not in RESERVED_COL_NAMES, \
        F"{name} is a reserved column name"

    return True


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
    assert_valid_name(column_name)
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


def get_type_by_name(db, name):
    assert_valid_name(name)
    c = db.cursor()
    c.execute(F"SELECT type from dictionary where name = '{name}'")
    row = c.fetchone()
    result = row[0] if row else None
    c.close()
    return result


def search_dictionary_for_type(db, object_name):
    c = db.cursor()
    c.execute(F"SELECT name from dictionary where type = '{object_name}'")
    rows = c.fetchall()
    results = []
    for row in rows:
        results.append(row[0])
    c.close()
    return sorted(results)


def add_thing(db, thing_type, thing_name, field_type):
    assert_valid_name(thing_name)
    t = get_type_by_name(db, thing_name)
    assert t is None or t == thing_type, \
        F"Conflicts with {t} also named '{thing_name}'"
    if not column_exists(db, "samples", thing_name):
        c = db.cursor()
        c.execute(
            F"""ALTER TABLE samples
            ADD COLUMN \"{thing_name}\" {field_type}"""
        )

        c.execute("INSERT OR REPLACE INTO dictionary(name, type) VALUES(?,?)",
                  (thing_name, thing_type))
        c.close()
        db.commit()
        return True
    return False


def del_thing(db, thing_type, thing_name):
    assert_valid_name(thing_name)
    t = get_type_by_name(db, thing_name)
    assert t is None or t == thing_type, \
        F"""No {thing_type} named '{thing_name}', but {t} exists with this name.
        Did you intent to call drop{t.capitalize()}() ?"""
    if (column_exists(db, "samples", thing_name)):
        rows = list_columns(db, "samples")
        cols = []
        typed_cols = []
        for row in rows:
            name, type, pk = row
            if (name != thing_name):
                quoted_name = '"' + name + '"'
                cols.append(quoted_name)
                tc = F"{quoted_name} {type}{' PRIMARY KEY' if pk else ''}"
                typed_cols.append(tc)

        str_cols = ",".join(cols)
        str_typed_cols = ",".join(typed_cols)

        c = db.cursor()
        script = F"""
            PRAGMA foreign_keys=off;
            BEGIN TRANSACTION;
            CREATE TABLE IF NOT EXISTS temp_samples ({str_typed_cols});
            INSERT INTO temp_samples({str_cols})
                SELECT {str_cols} FROM samples;
            DROP TABLE samples;
            ALTER TABLE temp_samples RENAME TO samples;
            DELETE FROM dictionary where name = '{thing_name}';
            COMMIT;
            PRAGMA foreign_keys=on;
            """
        c.executescript(script)
        c.close()
        db.commit()
        return True
    return False


def set_thing(db, thing_type, thing_name, method):
    assert_valid_name(thing_name)
    t = get_type_by_name(db, thing_name)
    assert t == thing_type, \
        F"No {thing_type} named {thing_name}"

    records = method(db, thing_name)

    # DEBUG TRACE
    # if thing_type == 'feature':
    #     for i, record in enumerate(records):
    #         print(i, '->', record)
    #         time.sleep(0.01)

    c = db.cursor()

    sql = F"""
        INSERT INTO samples (rowid, \"{thing_name}\") VALUES(?1,?2)
        ON CONFLICT(rowid) DO UPDATE SET \"{thing_name}\" = ?2
        """
    c.executemany(sql, records)
    affected_rows = c.rowcount
    c.close()
    db.commit()

    return affected_rows


def get_thing(db, thing_type, thing_name):
    assert_valid_name(thing_name)
    t = get_type_by_name(db, thing_name)
    assert t is None or t == thing_type, \
        F"""No {thing_type} named '{thing_name}', but {t} exists with this name.
        Did you intent to call get{t.capitalize()}() ?"""

    c = db.cursor()
    c.execute(F"SELECT \"{thing_name}\" from samples")
    rows = c.fetchall()
    results = []
    for row in rows:
        results.append(row[0])
    c.close()
    return results


def list_thing_values(db, thing_type, thing_name):
    assert_valid_name(thing_name)
    t = get_type_by_name(db, thing_name)
    assert t is None or t == thing_type, \
        F"""No {thing_type} named '{thing_name}', but {t} exists with this name.
        Did you intent to call get{t.capitalize()}() ?"""

    c = db.cursor()
    c.execute(F"SELECT distinct \"{thing_name}\" from samples")
    rows = c.fetchall()
    results = []
    for row in rows:
        results.append(row[0])
    c.close()
    return results


def list_column_values(db, col_name, cond=None):
    sql = F"SELECT distinct \"{col_name}\" from samples"
    if cond:
        sql += " WHERE " + cond

    c = db.cursor()
    c.execute(sql)
    rows = c.fetchall()

    results = []
    for row in rows:
        results.append(row[0])

    c.close()
    return results
