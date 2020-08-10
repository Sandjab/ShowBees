"""Dataset
Contains methods for audio datasets creation and manipulation
"""

# ===== Standard imports
import warnings
import os
from pathlib import Path
import sqlite3
import multiprocessing
from functools import partial
from contextlib import closing
import re

# ===== 3rd party imports
import pandas as pd
from checksumdir import dirhash
import librosa

# ===== Local imports
from lib.jupytools import mooltipath, iprint
from lib import dblib

# from .features import extract_feature_from_sample, welch, mfcc

# Disable warnings
warnings.filterwarnings('ignore')

# Note: "Public" instance or class methods consistently follow lower camelCase
#        naming convention, which does not comply with PEP 8.
#       "Private" methods do.
#        This is my personnal preference.
#        Live with it.


# ===== Main Class
class AudioDataset:
    RESERVED_COL_NAMES = ('row_id', 'sample_id')

    def __init__(self, dataset_name, source_path_str=None, nprocs=1):
        """Create an AudioDataset instance by parsing its manifest file

        Args:
            dataset_name (str): The name of the dataset
            source_path (Path): Path where to find the audio source files

        """

        # wherever you call from, the dataset root path
        # will be the same, \\|// relative to project root,
        # as marked by the -(@ @)- .kilroy flag file
        #            --oOO---(_)--- OOo--
        base_path = mooltipath("datasets")

        # dataset directory path
        self.path = Path(base_path, dataset_name)

        # database file path
        self.db_path = Path(self.path, "database.db")

        # manifest file path
        mnf_path = Path(base_path, dataset_name + ".mnf")

        # samples audio files path
        self.samples_path = Path(self.path, "samples")

        # if no source path was provided, it means
        # that we want to retrieve an existing AudioDataset db
        if not source_path_str:
            # if the database file does not exist, we bailout friendly
            if not self.db_path.is_file():
                iprint("Dataset database file does not exist.")
                iprint("Please insure that the dataset name is correct and")
                iprint("that the dataset has been previously created.")
                iprint("### ABORTING! ###")
                raise FileNotFoundError("Dataset database not found")

            # else retrieve dataset configuration from db
            self._get_config_from_db()
        else:
            # Else we have to build the dataset from its manifest.
            # we expect:
            #  - the manifest file to exist,
            #  - the ds directory to not exist
            #  - the source_dir to exist.
            # Else exit friendly
            source_path = Path(source_path_str).absolute()

            if not mnf_path.is_file():
                iprint("Dataset manifest does not exist.")
                iprint("Please insure that the dataset name is correct.")
                iprint("### ABORTING! ###")
                raise FileNotFoundError("Dataset manifest not found")

            if not source_path.is_dir():
                iprint(f"Source directory ({source_path}) does not exist.")
                iprint("Please insure that provided path is correct.")
                iprint("### ABORTING! ###")
                raise FileNotFoundError("Dataset source directory not found")

            if self.path.exists():
                iprint(f"The dataset directory ({self.path}) already exists.")
                iprint(
                    "If you really intent to CREATE this dataset, please "
                    "erase this directory first"
                )
                iprint("### ABORTING! ###")
                raise FileExistsError("Dataset directory already exists")

            iprint(f">>>>> Starting Dataset {dataset_name} build")

            # Create directory structure
            self.samples_path.mkdir(parents=True)

            # Read manifest
            with mnf_path.open("r") as f:
                lines = f.read().split("\n")
                sample_rate = int(lines[0])
                duration = float(lines[1])
                overlap = float(lines[2])
                md5 = lines[3]
                filenames = lines[4:]

            # Perform sanity checks
            assert duration > 0, "duration must be strictly positive"
            assert overlap >= 0, "overlap must be positive or zero"
            assert (
                sample_rate <= 44100 and sample_rate >= 1024
            ), "sample_rate must belong to [1024 - 44100]"
            assert overlap < duration, "overlap must be < duration"

            # Pre Populate instance properties
            self.sample_rate = sample_rate
            self.duration = duration
            self.overlap = overlap
            self.filenames = filenames
            self.nb_samples = 0

            # Create database tables
            dblib.create(self.db_path)

            # Perform actual audio chunks slicing
            self._build(source_path, nprocs)

            # Compute md5 checksum and check it agains manifest expected value
            assert self._is_valid(
                md5
            ), "Computed MD5 checksum does not match manifest expected value."
            iprint("Checksum OK!")

            # Everything went well, we are done (Yay!)
            iprint(f">>>>> Dataset {dataset_name} successfully created.")

        return

    def info(self):
        iprint("------------------------------------------------------")
        iprint("DATASET PATH          :", self.path)
        iprint("DATASET DB PATH       :", self.db_path)
        iprint("DATASET SAMPLES PATH  :", self.samples_path)
        iprint("NB SOURCE AUDIO FILES :", len(self.filenames))
        iprint("SAMPLE RATE           :", self.sample_rate)
        iprint("DURATION              :", self.duration)
        iprint("OVERLAP               :", self.overlap)
        iprint("NB AUDIO CHUNKS       :", self.nb_samples)
        iprint("------------------------------------------------------")
        return

    def _get_chunk_path(self, fi, ci):
        return Path(self.samples_path,
                    self._get_chunk_name(fi, ci) + ".wav")

    def _get_chunk_name(self, fi, ci):
        # chunk name is file id + chunk id left padded with 0s
        return str(fi).zfill(2) + "-" + str(ci).zfill(6)

    def _get_config_from_db(self):
        db = sqlite3.connect(self.db_path)
        # Indicates we want search results as dictionnaries
        db.row_factory = sqlite3.Row
        c = db.cursor()
        c.execute("SELECT * from config")
        row = dict(c.fetchone())
        self.sample_rate = row["sample_rate"]
        self.duration = row["duration"]
        self.overlap = row["overlap"]
        self.nb_samples = row["nb_samples"]
        db.row_factory = None
        c = db.cursor()
        c.execute("SELECT * from filenames")
        self.filenames = c.fetchall()
        db.close()
        return

    def _is_valid(self, expected_md5):
        """
        """
        # Compute md5 checksum over the dataset's samples directory
        iprint("Please wait, computing checksum...")
        computed_md5 = dirhash(self.samples_path, "md5")
        iprint("  Computed checksum", computed_md5)

        # No md5 in manifest means that this is the first build from
        # a potential provider. We display the md5 and remind user
        # to update the manifest
        if not expected_md5:
            iprint("  Manifest did not mention any checksum")
            iprint("  Probably because you are the dataset manifest creator")
            iprint("  Don't forget to update it with the above checksum.")
            iprint(
                "  Thus, when sharing this manifest with others,"
                " they can check for replicability"
            )
            return True
        else:
            iprint("  Expected checksum", expected_md5)
            return computed_md5 == expected_md5

    def _slice_one_file(self, file_idx, filename, source_path):
        sample_records = []

        iprint("[" + str(file_idx + 1) + "] " + filename)

        # load full sound file at once
        # Note: for some reason librosa.load does not accept 'Path',
        # even if it is supposed to, so we use string
        # TODO: Investigate this
        source, sr = librosa.core.load(
            os.fspath(source_path) + os.sep + filename, sr=self.sample_rate
        )

        # compute maximum number of chunks fitting in source duration
        dur = self.duration
        ovl = self.overlap
        n = int((len(source) - sr * ovl) / (dur * sr - ovl * sr))

        # Get one chunk of "duration" seconds at a time
        for chunk_idx in range(n):
            # Slice chunk data
            c_start_t = chunk_idx * (self.duration - self.overlap)
            c_end_t = c_start_t + self.duration
            c_data = source[int(c_start_t * sr): int(c_end_t * sr)]

            # Save chunk as audio file
            librosa.output.write_wav(
                self._get_chunk_path(file_idx, chunk_idx), c_data, sr
            )

            # Append record for later multiple db inserts
            sample_records.append(
                {
                    "name": self._get_chunk_name(file_idx, chunk_idx),
                    "file_id": file_idx + 1,
                    "start_t": c_start_t,
                    "end_t": c_end_t
                }
            )

        return sample_records

    def _build(self, source_path, nprocs):
        # We have to perform the task. Announce planned work
        nb_files = len(self.filenames)
        iprint(f"Ready to process {nb_files} audio files.")

        # Reset global record list
        total_records = []

        # if not multiprocessing, perform sequential processing
        if nprocs <= 1:
            # Walk the files list, perform slicing, compute perturbation ratio
            # and update database accordingly
            for file_idx, filename in enumerate(self.filenames):
                # slice one file and append records to the global list
                sample_records = self._slice_one_file(
                    file_idx, filename, source_path)

                total_records += sample_records
        # Else run a pool of workers processes
        else:
            with multiprocessing.Pool(nprocs) as pool:
                pool_records = pool.starmap(
                    partial(self._slice_one_file, source_path=source_path),
                    enumerate(self.filenames)
                )

                for sample_records in pool_records:
                    total_records += sample_records

        # Connect to the database, which will hold all the detailed
        # informations about the dataset
        iprint("Creating Database")
        db = sqlite3.connect(self.db_path)

        # Insert all file's chuncks records a once
        c1 = db.cursor()
        c1.executemany(
            "INSERT INTO samples "
            + "(name, file_id, start_t, end_t) "
            + "VALUES (:name, :file_id, :start_t, :end_t)",
            total_records
        )

        # Insert name of the source files we juste processed
        c2 = db.cursor()
        for f in self.filenames:
            c2.execute("INSERT INTO filenames (name) VALUES (?)", (f,))

        self.nb_samples = len(total_records)

        # Save configuration to db
        c3 = db.cursor()
        c3.execute(
            "INSERT INTO config (sample_rate, duration, overlap, nb_samples) "
            + "VALUES (?,?,?,?)",
            (self.sample_rate, self.duration, self.overlap, self.nb_samples),
        )

        # Commit and Close connection to db
        db.commit()
        db.close()
        iprint("Database created")
        return

    def query(self, sql, *args, as_dict=False):
        with sqlite3.connect(self.db_path) as db:
            if as_dict:
                db.row_factory = sqlite3.Row
            c = db.cursor()
            c.execute(sql, *args)
            rows = c.fetchall()
            for row in rows:
                if as_dict:
                    row = dict(row)
                print(row)

        db.close()
        return

    def _assert_valid_name(self, table_name, name):
        assert re.match("^[a-z0-9_]+$", name), \
            F"Invalid name ({table_name} names " \
            "must contain only lower case letters, digits and underscore)"
        assert name not in self.RESERVED_COL_NAMES, \
            F"{name} is a reserved column name"

        return True

    def _add_thing(self, table_name, name, type):
        self._assert_valid_name(table_name, name)
        with closing(sqlite3.connect(self.db_path)) as db:
            dblib.add_column(db, table_name, name, type)

    def _drop_thing(self, table_name, name):
        self._assert_valid_name(table_name, name)
        with closing(sqlite3.connect(self.db_path)) as db:
            dblib.del_column(db, table_name, name)

    def _list_things(self, table_name):
        with closing(sqlite3.connect(self.db_path)) as db:
            res = dblib.list_column_names(
                db,
                table_name,
                but=self.RESERVED_COL_NAMES)
            return sorted(res)

    def _exists_thing(self, table_name, name):
        self._assert_valid_name(table_name, name)
        with closing(sqlite3.connect(self.db_path)) as db:
            return dblib.column_exists(db, table_name, name)

    def _set_thing(self, table_name, name, method):
        self._assert_valid_name(table_name, name)
        with closing(sqlite3.connect(self.db_path)) as db:
            assert dblib.column_exists(db, table_name, name), \
                F"{name} is not a column in {table_name} table"

            records = method(db, name)
            c = db.cursor()

            sql = F"""
                INSERT INTO {table_name}(sample_id, {name}) VALUES(?1,?2)
                ON CONFLICT(sample_id) DO UPDATE SET {name} = ?2
                """
            c.executemany(sql, records)
            affected_rows = c.rowcount
            c.close()
            db.commit()

        return affected_rows

    def _get_thing_serie(self, table_name, names):
        assert len(names) > 0, "Missing arguments (names)"
        assert all(self._assert_valid_name(table_name, name) for name in names)

        with closing(sqlite3.connect(self.db_path)) as db:
            sql = F"SELECT {', '.join(names)} from {table_name}"
            c = db.cursor()
            iprint(sql)
            c.execute(sql)
            rows = c.fetchall()
            return rows

    def existsLabel(self, name):
        return self._exists_thing("labels", name)

    def addLabel(self, name):
        """add an new unpopulated label column to the labels table

        Args:
            name ([string]): [Name the new label]

        Returns:
            True if new column was added
            False if nothing was done because the column was alredy present

        Raises:
            Error if something else went wrong
        """
        self._add_thing("labels", name, "REAL")

    def setLabel(self, name, labelizer):
        return self._set_thing("labels", name, labelizer)

    def dropLabel(self, name):
        self._drop_thing("labels", name)

    def listLabels(self):
        return self._list_things("labels")

    def getLabelSerie(self, *names):
        return self._get_thing_serie("labels", names)

    def addAttribute(self, name):
        self._add_thing("attributes", name, "TEXT")

    def setAttribute(self, name, attributor):
        return self._set_thing("attributes", name, attributor)

    def dropAttribute(self, name):
        self._drop_thing("attributes", name)

    def listAttributes(self):
        return self._list_things("attributes")

    def getAttributeSerie(self, *names):
        return self._get_thing_serie("labels", names)

    def requestDataFrame(self, sql):
        with closing(sqlite3.connect(self.db_path)) as db:
            df = pd.read_sql_query(sql, db)
        return df

    def getDataFrame(self):
        sql = """
            SELECT * from samples s, attributes a, labels l
            where s.rowid = a.sample_id
            and s.rowid = l.sample_id
            """

        return self.requestDataFrame(sql).drop(columns=['sample_id'])
