import sqlite3
from lib.jupytools import iprint
import librosa
from pathlib import Path
import os


def load_lab_file(source_path, base_str):
    lab_path = Path(source_path, base_str + ".lab")
    assert lab_path.is_file(), 'File "' + base_str + '.lab"  no found'

    results = []
    with lab_path.open("r") as lab_file:
        lines = lab_file.read().split("\n")
        for line in lines:
            # strip lines with title, '.', or empty line
            if (line == base_str) or (line == ".") or (not line):
                continue

            results.append(line)

    return results


def slice(ds, source_path, th):
    """Build dataset chunks

            Args:
                None

            Returns:
                None
        """

    # We have to perform the task. Announce planned work
    nb_files = len(ds.filenames)
    iprint("Ready to process " + str(nb_files) + " audio files.")

    # Reset global chunk index
    sample_idx = 0

    # Connect to the database, which will hold all the detailed
    # informations about the dataset
    db = sqlite3.connect(ds.db_path)

    # Walk the files list, perform slicing, compute perturbation ratio
    # and update database accordingly
    for file_idx, filename in enumerate(ds.filenames):
        records = []
        iprint("[" + str(file_idx + 1) + "] " + filename)

        # strip file extension to get a base name
        base_str = str(Path(filename).stem)

        # load full sound file at once
        # Note: for some reason librosa.load does not accept 'Path',
        # even if it is supposed to, so we use string
        source, sr = librosa.core.load(
            os.fspath(source_path) + os.sep + filename, sr=ds.sample_rate
        )

        # load corresponding lab file
        lab_lines = load_lab_file(source_path, base_str)
        # lab_line_idx = 0
        # compute maximum number of chunks fitting in source duration
        n = int((len(source) - sr * ds.overlap) /
                (ds.duration * sr - ds.overlap * sr))

        # Get one chunk of "duration" seconds at a time
        for chunk_idx in range(n):
            # Slice chunk data
            c_start_t = chunk_idx * (ds.duration - ds.overlap)
            c_end_t = c_start_t + ds.duration
            c_data = source[int(c_start_t * sr): int(c_end_t * sr)]

            # Save chunk as audio file
            librosa.output.write_wav(ds.get_chunk_path(sample_idx), c_data, sr)

            # reset nobee chunk total time
            sum_nobee = 0.0

            # and compute chunk perturbation ratio
            for line_idx, line in enumerate(lab_lines):
                fields = line.split("\t")
                assert len(fields) == 3, (
                    "Invalid lab file line " +
                    str(line_idx + 1) + "(" + line + ")"
                )

                # label type (bee or no bee)
                label = fields[2]
                # label start en end times
                tp0 = float(fields[0])
                tp1 = float(fields[1])
                # skip annotation ending before chunk start
                if tp1 < c_start_t:
                    # lab_line_idx
                    continue

                # break on annotation starting after chunk end,
                # we are done.
                if tp0 > c_end_t:
                    break

                # only consider nobee intervals longer than threshold
                # to increment nobee sum
                if label == "nobee" and (tp1 - tp0) >= th:
                    # compute segment and annotation overlap.
                    sum_nobee += min(tp1, c_end_t) - max(c_start_t, tp0)

            # Perturbation ratio is the proportion of nobee in the
            # chunk, rounded to 3 decimals.
            p_ratio = round(sum_nobee / ds.duration, 3)

            # Append record for later multiple db inserts
            records.append(
                {
                    "id": sample_idx,
                    "hive": filename[:5],
                    "start_t": c_start_t,
                    "end_t": c_end_t,
                    "p_ratio": p_ratio,
                }
            )

            # increment total number of chunks in dataset
            sample_idx += 1

        # Insert all file's chuncks records a once
        c1 = db.cursor()
        c1.executemany(
            "INSERT INTO samples (id, hive, start_t, end_t, p_ratio) VALUES (:id, :hive, :start_t, :end_t, :p_ratio)",
            records,
        )

        # Insert name of the source file we juste processed
        c2 = db.cursor()
        c2.execute("INSERT INTO filenames (name) VALUES (?)", (filename,))
        db.commit()

    ds.nb_samples = sample_idx

    # Save configuration to db
    c3 = db.cursor()
    c3.execute(
        "INSERT INTO config (sample_rate, duration, overlap, nb_samples) VALUES (?,?,?,?)",
        (ds.sample_rate, ds.duration, ds.overlap, ds.nb_samples),
    )
    db.commit()
    # Close connection to db
    db.close()

    return
