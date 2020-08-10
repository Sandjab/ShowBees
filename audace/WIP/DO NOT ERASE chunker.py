"""
Contains functions for building datasets (audio chunks, threshold files, label files) from a dataset manifest
"""# ===== Standard imports
import os
import sys
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
#from features import extract_feature_from_chunk, welch, mfcc



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
    """Build chunk label, only for chunk i, threshold file"""

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

def build_dataset_projector_files(dataset_name, save_dir_str, th=0):
    output_path = Path(save_dir_str)

    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    dataset_path =  mooltipath('datasets', dataset_name)
    threshold_path = Path(dataset_path, 'thresholds', 'threshold_' + str(th) + '.csv')
    
    assert threshold_path.is_file(), 'No threshold file found for threshold value = ' + str(th)

    with threshold_path.open('r') as threshold_file:
        threshold_reader = csv.DictReader(threshold_file, delimiter=',')
        with Path(output_path, 'labels.tsv').open('w', newline='') as labels_file:
            with Path(output_path, 'tensors.tsv').open('w', newline='') as tensors_file:
                tensors_writer = csv.writer(tensors_file, delimiter='\t')

                for row in threshold_reader:
                    if row['label'] == 'bee':
                        label_state = get_hive_label_from_chunk_name(row['name'])
                        tensor = extract_feature_from_chunk(os.fspath(Path(dataset_path, 'chunks', row['name'] + '.wav')), welch)
                        tensors_writer.writerow(tensor)
                        labels_file.write(label_state+'\n')
                        
    return    




def get_list_samples_names(path_audioSegments_folder, extension='.wav'):
    print(path_audioSegments_folder)
    sample_ids = [os.path.basename(x) for x in glob.glob(
        os.path.join(path_audioSegments_folder,'*'+extension))]
    return sample_ids
