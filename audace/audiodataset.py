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

# ===== 3rd party imports
import pandas as pd
from checksumdir import dirhash
import librosa
from tqdm.auto import tqdm
import soundfile

# ===== Local imports
from audace.jupytools import mooltipath, iprint
from audace import dblib

# from .features import extract_feature_from_sample, welch, mfcc

# Disable warnings
warnings.filterwarnings('ignore')

# Note: "Public" instance or class methods consistently follow lower camelCase
#        naming convention, which does not comply with PEP 8.
#       "Private" methods do.
#        This is my personnal preference.
#        Deal with it.


# ===== Main Class
class AudioDataset:
    def __init__(self, dataset_name, source_path_str=None, nprocs=1):
        """Create an AudioDataset instance by parsing its manifest file

        Args:
            dataset_name (str): The name of the dataset
            source_path (Path): Path where to find the audio source files

        """

        self.ds_name = dataset_name

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
            self._is_valid(md5)
            # assert self._is_valid(
            #     md5
            # ),"Computed MD5 checksum does not match manifest expected value."
            # iprint("Checksum OK!")

            # Everything went well, we are done (Yay!)
            iprint(f">>>>> Dataset {dataset_name} successfully created.")

        return

    def _cnx(self):
        return closing(
            sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        )

    def info(self):
        iprint("------------------------------------------------------")
        iprint("DATASET NAME          :", self.ds_name)
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
        self.ds_name = row["ds_name"]
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
            soundfile.write(
                str(self._get_chunk_path(file_idx, chunk_idx)),
                c_data,
                sr
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
        iprint(f"Starting to process {nb_files} audio files.")

        # Reset global record list
        total_records = []

        # total_pb = tqdm(desc="Total", position=0, total=nb_files)
        # process_pb = []
        # for i in range(nprocs):
        #     process_pb.append(tqdm(desc=F"Process {i:02d}"))

        # if not multiprocessing, perform sequential processing
        if nprocs <= 1:
            # Walk the files list, perform slicing, compute perturbation ratio
            # and update database accordingly
            for file_idx, filename in tqdm(enumerate(self.filenames),
                                           total=nb_files):
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
            "INSERT INTO config"
            "(ds_name, sample_rate, duration, overlap, nb_samples) "
            "VALUES (?,?,?,?,?)",
            (
                self.ds_name,
                self.sample_rate,
                self.duration,
                self.overlap,
                self.nb_samples
            )
        )

        # Commit and Close connection to db
        db.commit()
        db.close()
        iprint("Database created")
        return

    def query(self, sql, *args, as_dict=False):
        with self._cnx() as db:
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

    def addLabel(self, name):
        with self._cnx() as db:
            return dblib.add_thing(db, "label", name, "REAL")

    def setLabel(self, name, labelizer):
        with self._cnx() as db:
            return dblib.set_thing(db, "label", name, labelizer)

    def getLabel(self, name):
        with self._cnx() as db:
            return dblib.get_thing(db, "label", name)

    def dropLabel(self, name):
        with self._cnx() as db:
            return dblib.del_thing(db, "label", name)

    def listLabels(self):
        with self._cnx() as db:
            return dblib.search_dictionary_for_type(db, "label")

    def addAttribute(self, name):
        with self._cnx() as db:
            return dblib.add_thing(db, "attribute", name, "TEXT")

    def getAttribute(self, name):
        with self._cnx() as db:
            return dblib.get_thing(db, "attribute", name)

    def setAttribute(self, name, attributor):
        with self._cnx() as db:
            return dblib.set_thing(db, "attribute", name, attributor)

    def dropAttribute(self, name):
        with self._cnx() as db:
            return dblib.del_thing(db, "attribute", name)

    def listAttributes(self):
        with self._cnx() as db:
            return dblib.search_dictionary_for_type(db, "attribute")

    def addFeature(self, name):
        with self._cnx() as db:
            return dblib.add_thing(db, "feature", name, "feature")

    def setFeature(self, name, featurizer):
        with self._cnx() as db:
            return dblib.set_thing(db, "feature", name, featurizer)

    def getFeature(self, name):
        with self._cnx() as db:
            return dblib.get_thing(db, "feature", name)

    def dropFeature(self, name):
        with self._cnx() as db:
            return dblib.del_thing(db, "feature", name)

    def listFeatures(self):
        with self._cnx() as db:
            return dblib.search_dictionary_for_type(db, "label")

    def queryDataFrame(self, sql):
        with self._cnx() as db:
            df = pd.read_sql_query(sql, db)
        return df

    def dumpDataFrame(self):
        sql = "SELECT * from samples s"
        return self.queryDataFrame(sql)

    def getDataFrame(self):
        return self.dumpDataFrame.drop(columns=['XXX'])
