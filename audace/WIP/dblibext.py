import sqlite3


class DbCmd:
    DELETE = 'DELETE FROM {}'
    DROP = 'DROP TABLE IF EXISTS {}'
    COUNT = 'SELECT COUNT(*) FROM ?'
    INIT = """CREATE TABLE IF NOT EXISTS config(
                    sample_rate INT NOT NULL,
                    duration REAL NOT NULL,
                    overlap REAL NOT NULL,
                    nb_samples INT NOT NULL)"""


class SafeCursor:
    def __init__(self, cnx):
        self._cnx = cnx
        self._cursor = self._cnx.cursor()

    def __enter__(self):
        return self._cursor

    def __exit__(self, typ, value, traceback):
        self._cnx.row_factory = None
        self._cursor.close()


class DatasetDb:

    # Normal constructor
    def __init__(self, path):
        self._path = path
        self._cnx = None

    # Constructor when created via "with"
    def __enter__(self):
        if not self._cnx:
            self._cnx = sqlite3.connect(self._path)
        return self

    # "with" epilog
    def __exit__(self, typ, value, traceback):
        self._cnx.commit()
        self._cnx.close()
        self._cnx = None

    def commit(self):
        self._cnx.commit()

    def rollback(self):
        self._cnx.rollback()

    def nuke(self):
        # Reset Database
        with SafeCursor(self._cnx) as c:
            c.execute("DROP TABLE IF EXISTS SAMPLES")
            c.execute("DROP TABLE IF EXISTS LABELS")
            c.execute("DROP TABLE IF EXISTS FILENAMES")
            c.execute("DROP TABLE IF EXISTS FEATURES")

        self.commit()

        return

    def execute(self, c, sql, *args):
        if not args:
            return c.execute(sql)

        return c.execute(sql, args)

    def getCount(self, table_name, cond='', *args):
        req = f"SELECT COUNT(*) FROM '{table_name}'"
        if cond:
            req += " WHERE " + cond
        with SafeCursor(self._cnx) as c:
            return self.execute(c, req, *args).fetchone()[0]

    def getOneAsList(self, table_name, cond='', *args):
        req = f"SELECT * FROM '{table_name}'"
        if cond:
            req += " WHERE " + cond

        with SafeCursor(self._cnx) as c:
            return self.execute(c, req, *args).fetchone()

    def getOneAsDict(self, table_name, cond='', *args):
        req = f"SELECT * FROM '{table_name}'"
        if cond:
            req += " WHERE " + cond

        self._cnx.row_factory = sqlite3.Row
        with SafeCursor(self._cnx) as c:
            return dict(self.execute(c, req, *args).fetchone())

    def getAllAsList(self, table_name, cond='', *args):
        req = f"SELECT * FROM '{table_name}'"
        if cond:
            req += " WHERE " + cond

        with SafeCursor(self._cnx) as c:
            return self.execute(c, req, *args).fetchall()

    def getAllAsDict(self, table_name, cond='', *args):
        req = f"SELECT * FROM '{table_name}'"
        if cond:
            req += " WHERE " + cond

        self._cnx.row_factory = sqlite3.Row
        with SafeCursor(self._cnx) as c:
            rows = self.execute(c, req, *args).fetchall()
            return [dict(row) for row in rows]
