"""Dataset
Contains method for building datasets
"""  # ===== Standard imports
import warnings
import os
from pathlib import Path
import csv
import sqlite3
import multiprocessing
from functools import partial

# ===== 3rd party imports
from checksumdir import dirhash
import librosa

# ===== Local imports
from lib.jupytools import mooltipath, iprint
from lib.dblib import DatasetDb

# from .features import extract_feature_from_sample, welch, mfcc

# Disable warnings
warnings.filterwarnings('ignore')


# ===== Main Class
class AudioDataset:
    def __init__(self, dataset_name, source_path_str=None, th=0, nprocs=1):
        """Create an AudioDataset instance by parsing its manifest file

        Args:
            dataset_name (str): The name of the dataset
            source_path (Path): Path where to find the audio and
            lab source files

        Returns:
            a Dataset instance with the populated instance's properties
            - **path* (Path) : Path to the dataset root directory
            - **samples_path* (Path) : Path to the dataset audio
                                       chunks subdirectory
            - **source_path** (Path) : Path where to find the audio and
                                       lab source files
            - **filenames** (tuple) : The list of source audio filenames
                                      to be chunked
            - **sample_rate** (*int*) : The audio chunks target sample rate
                                        (used during resampling)
            - **duration** (*float*) : The audio chunks duration
            - **overlap** (*float*) : The audio chunks overlap
            - **nb_chuncks* (*int*) : The number of actually generated chunks
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
            self._create_tables()

            # Perform actual audio chunks slicing
            self._build(source_path, th, nprocs)

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

    # TODO: This is a WIP, disregard for now
    def _get_config_from_db_candidate(self):
        with DatasetDb(self.db_path) as db:
            row = db.GetOneAsDict("SELECT * from config")
            self.sample_rate = row["sample_rate"]
            self.duration = row["duration"]
            self.overlap = row["overlap"]
            self.nb_samples = row["nb_samples"]

            self.filenames = db.getAllAsList("SELECT * from filenames")
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

    def _load_lab_file(self, source_path, base_str):
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

    def _slice_one(self, file_idx, filename, source_path, th):
        sample_records = []

        # strip file extension to get a base name
        base_str = str(Path(filename).stem)

        iprint("[" + str(file_idx + 1) + "] " + filename)

        # load full sound file at once
        # Note: for some reason librosa.load does not accept 'Path',
        # even if it is supposed to, so we use string
        # TODO: Investigate this
        source, sr = librosa.core.load(
            os.fspath(source_path) + os.sep + filename, sr=self.sample_rate
        )

        # load corresponding lab file
        lab_lines = self._load_lab_file(source_path, base_str)

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

            # reset nobee chunk total time
            sum_nobee = 0.0

            # and compute chunk perturbation ratio
            for li, line in enumerate(lab_lines):
                fields = line.split("\t")
                assert len(fields) == 3, (
                    "Invalid lab file line " + str(li + 1) + "(" + line + ")"
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
            p_ratio = round(sum_nobee / self.duration, 3)

            # Append record for later multiple db inserts
            sample_records.append(
                {
                    "name": self._get_chunk_name(file_idx, chunk_idx),
                    "file_id": file_idx + 1,
                    "hive": filename[:5],
                    "start_t": c_start_t,
                    "end_t": c_end_t,
                    "p_ratio": p_ratio,
                }
            )

        return sample_records

    def _build(self, source_path, th, nprocs):
        """Build dataset chunks

            Args:
                None

            Returns:
                None
        """

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
                sample_records = self._slice_one(
                    file_idx, filename, source_path, th)

                total_records += sample_records
        # Else run a pool of workers processes
        else:
            with multiprocessing.Pool(nprocs) as pool:
                pool_records = pool.starmap(
                    partial(self._slice_one, source_path=source_path, th=th),
                    enumerate(self.filenames)
                )

                for sample_records in pool_records:
                    total_records += sample_records

        # for r in records:
            # iprint(r)

        # Connect to the database, which will hold all the detailed
        # informations about the dataset
        iprint("Creating Database")
        db = sqlite3.connect(self.db_path)

        # Insert all file's chuncks records a once
        c1 = db.cursor()
        c1.executemany(
            "INSERT INTO samples "
            + "(name, file_id, hive, start_t, end_t, p_ratio) "
            + "VALUES (:name, :file_id, :hive, :start_t, :end_t, :p_ratio)",
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

    def _create_tables(self):
        iprint("Creating database tables")
        with sqlite3.connect(self.db_path) as db:
            c = db.cursor()
            # Create table to store AudioDataset configuration
            c.execute(
                """CREATE TABLE IF NOT EXISTS config(
                    sample_rate INT NOT NULL,
                    duration    REAL NOT NULL,
                    overlap     REAL NOT NULL,
                    nb_samples  INT NOT NULL)"""
            )

            # Create table to store AudioDataset list of source files names
            c.execute("CREATE TABLE IF NOT EXISTS filenames(name TEXT)")

            # Create table to store AudioDataset samples information
            c.execute(
                """CREATE TABLE IF NOT EXISTS samples(
                    parent_id   INT DEFAULT 0,
                    name        TEXT,
                    file_id     INT,
                    augment_id  INT,
                    hive        TEXT,
                    start_t     REAL,
                    end_t       REAL,
                    p_ratio     REAL)"""
            )

            # Create table to store augmentation information
            c.execute(
                """CREATE TABLE IF NOT EXISTS augmentations(
                    method          TEXT NOT NULL,
                    int_param1      INT,
                    int_param2      INT,
                    float_param1    REAL,
                    float_param2    REAL)"""
            )

            # Create index on augmentations
            c.execute(
                """CREATE UNIQUE INDEX augm_idx ON augmentations (
                    sample_id,
                    name) """
            )

            # Create table to store samples attributes
            c.execute(
                """CREATE TABLE IF NOT EXISTS attributes(
                    sample_id INT,
                    str_value TEXT)"""
            )

            # Create index on attributes
            c.execute(
                """CREATE UNIQUE INDEX attr_idx ON attributes (
                    sample_id,
                    name)"""
            )

            # Create table to store AudioDataset samples labels
            c.execute(
                """CREATE TABLE IF NOT EXISTS labels(
                    sample_id   INT,
                    name        TEXT,
                    str_value   TEXT,
                    int_value   INT,
                    strength    REAL)"""
            )

            # Create index on labels
            c.execute(
                """CREATE UNIQUE INDEX lbl_idx ON labels (
                    sample_id,
                    name) """
            )

            # Create table to store AudioDataset samples features
            c.execute(
                """CREATE TABLE IF NOT EXISTS features(
                    sample_id INT,
                    code TEXT,
                    x INT,
                    y INT,
                    json TEXT,
                    str_value TEXT)"""
            )

            # Create index on features
            c.execute(
                """CREATE INDEX ftr_idx ON features (
                sample_id,
                code) """
            )

            db.commit()

        db.close()
        return

    def csvdump(self, output_path, sep=","):
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            c = db.cursor()
            c.execute("SELECT * FROM samples")
            rows = c.fetchall()
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=rows[0].keys(), delimiter=sep)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))
        db.close()
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

    def search(self, cond, *args, as_dict=False):
        with sqlite3.connect(self.db_path) as db:
            if as_dict:
                db.row_factory = sqlite3.Row
            c = db.cursor()
            c.execute("SELECT * FROM SAMPLES WHERE " + cond, *args)
            rows = c.fetchall()
            for row in rows:
                if as_dict:
                    row = dict(row)
                print(row)

        db.close()

        return
