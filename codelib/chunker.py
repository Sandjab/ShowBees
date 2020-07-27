# ===== Standard imports
import os
import glob
import shutil
from pathlib import Path
import csv
import re

# ===== 3rd party imports
from checksumdir import dirhash
import librosa
import numpy as np

# ===== Local imports
from jupytools import mooltipath
from info import printb, printr, printp, print

def load_dataset_manifest(dataset_name):
    """Load a dataset manifest

    Parses a manifest file and return dataset parameters.

    Args:
        dataset_name (str): The name of the dataset 

    Returns:
        (tuple): a 5 elements t-uple containing

        - **sample_rate** (*int*) : The audio chunks target sample rate (used during resampling)
        - **duration** (*float*) : The audio chunks duration
        - **overlap** (*float*) : The audio chunks overlap
        - **chuncks_md5** (*str*) : The expercted md5 checksum of the audio chunk directory
        - **filenames** (tuple) : The list of source audio filenames to be chunked
    """
    manifest = mooltipath('datasets', dataset_name + '.mnf')
    assert manifest.is_file(), 'Manifest not found for dataset ' + dataset_name + '.'

    with manifest.open('r') as f:
        lines = f.read().split('\n')
        sample_rate = int(lines[0])
        duration = float(lines[1])
        overlap = float(lines[2])
        chunks_md5 = lines[3]
        filenames = lines[4:]

    #Sanity checks
    assert duration > 0, 'duration must be strictly positive'
    assert overlap >= 0, 'overlap must be positive or zero'
    assert sample_rate <= 44100 and sample_rate >= 1024, 'sample_rate must belong to [1024 - 44100]'
    assert overlap < duration, 'overlap must be strictly less than duration'

    return filenames, sample_rate, duration, overlap, chunks_md5 


def build_dataset_labs(dataset_name, input_path):
    """Copy dataset .lab files from a source directory to the appropriate dataset directory
    
        Args:
            dataset_name (str): Name of the dataset
            input_path (str): Path to the source directory

        Returns:
            (int): Number of copied lab files
    """
    # Get needed dataset parameters from manifest file
    filenames, *_ = load_dataset_manifest(dataset_name)
        
    output_path = mooltipath('datasets', dataset_name, 'labs')
    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    # Copy .lab files from source to target directory
    for filename in filenames:
        base_name_str = str(Path(filename).stem)
        lab_full_path = Path(input_path, base_name_str + '.lab')
        assert lab_full_path.is_file(), 'Missing file ' + base_name_str + '.lab in ' + input_path
        shutil.copy(lab_full_path, output_path)

    return len(filenames)    


def build_dataset_chunks(dataset_name, input_path):
    """Build dataset chunks, from a dataset name

        Args:
            dataset_name (str): Name of the dataset
            input_path (str): Path to the source directory

        Returns:
            (tuple):
    """
    # Get all dataset parameters from manifest
    filenames, sample_rate, duration, overlap, chunks_md5 = load_dataset_manifest(dataset_name)
    
    # Create output directory, if needed
    output_path = mooltipath('datasets', dataset_name, 'chunks')
    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    nb_files = len(filenames)
    nb_chunks = 0

    # Announce planned work
    printb('Starting to process ' + str(nb_files) + ' audio files.')

    # Walk the file list
    for filename in filenames:
        print(filename)
        base_name_str = str(Path(filename).stem)
        chunk_manifest_path = Path(output_path, base_name_str + '.csv')

        with chunk_manifest_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'start_t', 'end_t'], delimiter=',')
            writer.writeheader()    

            # load full sound file at once
            source, sr = librosa.core.load(os.path.join(input_path, filename), sr=sample_rate)

            # compute maximum number of chunks fitting in source duration
            n = int( (len(source) - sr*overlap) / (duration*sr - overlap*sr))
        
            # if source file is shorter than duration, skip it
            if (n <= 0):
                continue
        
            # Get one chunk of "duration" seconds at a time
            for i in range(n):
                nb_chunks += 1
                chunk_name = base_name_str + '_chunk' + str(i).zfill(4)
                # Save chunk
                start_t = i*(duration-overlap)
                end_t = start_t + duration
                chunk = source[int(start_t*sr):int(end_t*sr)]
                librosa.output.write_wav(Path(output_path, chunk_name + '.wav'), chunk, sr)
                writer.writerow({'name': chunk_name, 'start_t': start_t, 'end_t': end_t})

    printb('Done')    
    
    # Compute md5 checksum and check it agains manifest expected value
    print('Computing checksum...')
    md5h = dirhash(output_path, 'md5')
    assert md5h == chunks_md5, 'MD5 checksum (' + md5h + ') does not match manifest value (' + chunks_md5 +').'
    print('Checksum OK!')
    return output_path, nb_files, nb_chunks, md5h 

def build_dataset_thresholds(dataset_name, thresholds):
    """Build dataset thresholds file, from a dataset name

        Args:
            dataset_name (str): Name of the dataset
            input_path (str): Path to the source directory

        Returns:
            (tuple):
    """
    # Get needed dataset parameters from manifest file
    filenames, sample_rate, duration, overlap, chunks_md5 = load_dataset_manifest(dataset_name)
    
    # Create output directory, if needed
    output_path = mooltipath('datasets', dataset_name, 'thresholds')
    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    # Generate threshold file for each threshold)
    for th in thresholds:
        threshold_file_path = Path(output_path, 'threshold_'+str(th) + '.csv')
        
        with threshold_file_path.open('w', newline='') as threshold_file:
            writer = csv.DictWriter(threshold_file, fieldnames=[
                            'name', 'start_t', 'end_t', 'strength', 'label'], delimiter=',')
            writer.writeheader()

            # Walk the source file list
            for filename in filenames:
                base_name_str = str(Path(filename).stem)
                print(base_name_str)
                lab_full_path = mooltipath('datasets', dataset_name, 'labs', base_name_str + '.lab')
                manifest_full_path = mooltipath('datasets', dataset_name, 'chunks', base_name_str + '.csv')

                # Iterate over the manifest
                with manifest_full_path.open('r') as manifest_file:
                    reader = csv.DictReader(manifest_file)
                    for row in reader:
                        chunk_name = row['name']
                        chunk_start_t = float(row['start_t'])
                        chunk_end_t = float(row['end_t'])
                        sum_nobee = 0.0
                        with lab_full_path.open('r') as lab_file:
                            lines = lab_file.read().split('\n')
                            for line in lines:
                                if (line == base_name_str) or (line == '.') or (line == ''):
                                    # ignores title, '.', or empty line on the file.
                                    continue

                                parsed_line = line.split('\t')

                                assert (len(parsed_line) == 3), ('expected 3 fields in each line, got: '+str(len(parsed_line)))

                                tp0 = float(parsed_line[0])
                                tp1 = float(parsed_line[1])
                                label = parsed_line[2]

                                # no need to read further annotation starting after chunk end
                                if tp0 > chunk_end_t:
                                    break

                                # skip annotation ending before chunk start 
                                if tp1 < chunk_start_t:
                                    continue

                                # only consider nobee intervals longer than threshold
                                if label == 'nobee' and (tp1 - tp0) >= th:
                                    sum_nobee = sum_nobee + min(tp1, chunk_end_t) - max(chunk_start_t, tp0)

                        if (sum_nobee > 0):
                            label_th = ["nobee", round(sum_nobee / (chunk_end_t - chunk_start_t), 3)]
                        else:
                            label_th = ['bee', 0.0]

                        writer.writerow({'name': chunk_name, 'start_t': chunk_start_t, 'end_t': chunk_end_t,
                                'strength': label_th[1],  'label': label_th[0]})
    printb('---------------------- Done ----------------------')

    return

def get_hive_label_from_chunk_name(filename):
    state_label = None
    # rule (patern, label), stricter first
    rules = [
        ('(?i)active', 'queen'), 
        ('(?i)missing queen', 'noqueen'),
        ('NO_QueenBee', 'noqueen'),
        ('QueenBee', 'queen'),       
    ]

    for rule in rules:
        pattern, label = rule
        if (re.search(pattern, filename)):
            state_label = label
            break

    assert   state_label, 'Unable to assign label to chunk ' + filename    
    return state_label


def build_dataset_labels(dataset_name, save_dir_str, th=0):
    liste = []
    output_path = Path(save_dir_str)

    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    threshold_path = mooltipath('datasets', dataset_name, 'thresholds', 'threshold_' + str(th) + '.csv')
    
    assert threshold_path.is_file(), 'No threshold file found for threshold value = ' + str(th)

    with threshold_path.open('r') as rfile:
        with Path(output_path, 'state_labels.csv').open('w', newline='') as wfile:
            reader = csv.DictReader(rfile, delimiter=',')
            writer = csv.DictWriter(wfile, fieldnames=['name', 'label'], delimiter=',')
            writer.writeheader()

            for row in reader:
                if row['label'] == 'bee':
                    label_state = get_hive_label_from_chunk_name(row['name'])
                    writer.writerow({'name': row['name'], 'label': label_state})
                else:
                    liste.append(row['name'])
    return liste




def get_list_samples_names(path_audioSegments_folder, extension='.wav'):
    print(path_audioSegments_folder)
    sample_ids = [os.path.basename(x) for x in glob.glob(
        os.path.join(path_audioSegments_folder,'*'+extension))]
    return sample_ids
