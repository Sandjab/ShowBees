from pathlib import Path


class Labelizer:
    def __init__(self, transformers):
        self._transformers = transformers

    def _process(self):
        return


class FromFileName(Labelizer):
    def __init__(self, F):
        self._F = F

    def __call__(self, db, label_name):
        c = db.cursor()
        sql = """SELECT s.rowid, F(f.name) FROM samples s
                 INNER JOIN filenames f
                 ON s.file_id = f.rowid"""

        db.create_function("F", 1, self._F)

        c.execute(sql)
        rows = c.fetchall()
        c.close()

        return rows


# TODO (or discard)
class FromTableColumn(Labelizer):
    def __init__(self, table, column):
        self._table = table
        self._column = column

    def __call__(self):
        return


class FromAnnotation(Labelizer):
    def __init__(self, path, th=0):
        self._path = path
        self._th = th

    def __call__(self, db, label_name):
        c = db.cursor()
        c.execute("SELECT rowid, name from filenames")
        file_rows = c.fetchall()
        c.close()

        records = []
        for file_row in file_rows:
            file_id, filename = file_row

            # strip file extension to get a base name
            base_str = str(Path(filename).stem)

            sql = """SELECT rowid, start_t, end_t FROM samples
                    WHERE file_id = ?"""
            c = db.cursor()
            c.execute(sql, (file_id,))
            sample_rows = c.fetchall()
            for sample_row in sample_rows:
                sample_id, start_t, end_t = sample_row

                # reset label chunk total time
                sum_label = 0.0

                # load corresponding lab file
                lab_lines = self._load_lab_file(self._path, base_str)

                # and compute label strength
                for lab_line in lab_lines:
                    lbl, tp0, tp1 = lab_line

                    # skip annotation ending before chunk start
                    if tp1 < start_t:
                        # lab_line_idx
                        continue

                    # break on annotation starting after chunk end,
                    # we are done.
                    if tp0 > end_t:
                        break

                    # only consider label intervals longer than
                    # unitary threshold to increment label sum
                    if lbl == label_name and (tp1 - tp0) >= self._th:
                        # compute segment and annotation overlap.
                        sum_label += min(tp1, end_t) - max(start_t, tp0)

                # strength is the proportion of label in the
                # chunk, rounded to 3 decimals.
                strength = round(sum_label / (end_t - start_t), 3)

                records.append(
                    (sample_id, strength))

        return records

    def _load_lab_file(self, source_path, base_str):
        lab_path = Path(source_path, base_str + ".lab")
        assert lab_path.is_file(), 'File "' + base_str + '.lab"  no found'

        results = []
        with lab_path.open("r") as lab_file:
            lines = lab_file.read().split("\n")
            for li, line in enumerate(lines):
                # strip lines with title, '.', or empty line
                if (line == base_str) or (line == ".") or (not line):
                    continue

                fields = line.split("\t")
                assert len(fields) == 3, (
                    "Invalid lab file line " + str(li + 1) + "(" + line + ")"
                )

                results.append((fields[2], float(fields[0]), float(fields[1])))

        return results
