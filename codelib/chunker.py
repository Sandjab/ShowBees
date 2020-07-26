# ===== Standard imports
import os
import glob
from pathlib import Path
import csv

# ===== 3rd party imports
from checksumdir import dirhash
import librosa
import numpy as np

# ===== Local imports
from jupytools import mooltipath
from info import printb, printr, printp, print


def save_labels(audiofilename, chunk_name,  chunk_start_t, chunk_end_t, annotations_path, threshold=0):
    chunk_length = chunk_end_t - chunk_start_t

    # to support special cases of hive naming...
    if audiofilename.startswith('#'):
        annotation_filename = audiofilename[1:-4]+'.lab'
    else:
        annotation_filename = audiofilename[0:-4]+'.lab'

    labels_th = ['bee', 0.0]

    with open(os.path.join(annotations_path, annotation_filename), 'r') as f:
        lines = f.read().split('\n')

        sum_nobee = 0
        for line in lines:
            if (line == annotation_filename[0:-4]) or (line == '.') or (line == ''):
                # ignores title, '.', or empty line on the file.
                continue

            parsed_line = line.split('\t')

            assert (len(parsed_line) ==
                    3), ('expected 3 fields in each line, got: '+str(len(parsed_line)))

            tp0 = float(parsed_line[0])
            tp1 = float(parsed_line[1])
            label = parsed_line[2]

            if tp0 > chunk_end_t:  # no need to read further annotation starting after chunk end
                break

            if tp1 < chunk_start_t:  # skip annotation ending before chunk start 
                continue

            # only consider nobee intervals longer than threshold
            if label == 'nobee' and (tp1 - tp0) >= threshold:
                sum_nobee = sum_nobee + min(tp1, chunk_end_t) - max(chunk_start_t, tp0)

        if (sum_nobee > 0):
            labels_th = ["nobee", round(sum_nobee/chunk_length, 3)]

    return labels_th


def dataset_dirname(sample_rate=22050, duration=1, overlap=0):
    return 'chunks_SR' + str(sample_rate) + "Hz_DUR" + str(duration) + 's_OVL' + str(overlap) + "s"
    
def build_dataset(dataset_name, input_path, duration=1, overlap=0, sample_rate=22050):
    """Build dataset chunks, from a dataset name"""
    manifest = mooltipath('datasets', 'MNF_' + dataset_name + '.lof')
    print("Looking for manifest ", manifest)
    assert manifest.is_file(), 'No manifest found for dataset ' + dataset_name

    with open(manifest, 'r') as f:
        filenames = f.read().split('\n')
        output_path = mooltipath('datasets', dataset_name, dataset_dirname(sample_rate, duration, overlap))
        _, nb_files, nb_chunks = build_chunks(input_path, output_path, duration, overlap, sample_rate, filenames)
        md5h = dirhash(output_path, 'md5')

    return output_path, nb_files, nb_chunks, md5h 

def build_chunks(input_path, output_path, duration=1, overlap=0, sample_rate=22050, filenames=None):
    """Slice all sound files (*.wav and *.mp3) within a directory into chunks, and assign labels to these chunks

    Extended description here

    Args:
        input_path (str):
        output_path (str):
        duration (int):
        overlap (int):
        sample_rate (int):

    Returns:
        nb_files (int):
        nb_chunks (int):
    """

    #Sanity check
    assert duration > 0, "duration must be strictly positive"
    assert overlap >= 0, "overlap must be positive or zero"
    assert sample_rate <= 44100 and sample_rate >= 1024, "sample_rate must belong to [1024 - 44100]"
    assert overlap < duration, "overlap must be strictly less than duration"

    if not filenames:
        # Build audio filenames list (both wav and mp3)
        paths = glob.glob(os.path.join(input_path, '*.mp3'))
        paths.extend(glob.glob(os.path.join(input_path, '*.wav'))) 
        filenames = [os.path.basename(x) for x in paths]

    # Create output directory, if needed
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    nb_files = len(filenames)
    nb_chunks = 0
    # Announce planned work
    printb("Processing "+str(nb_files) + " audio files.")

    # Walk the file list
    for filename in filenames:
        print(filename)

        # load full sound file at once
        source, sr = librosa.core.load(os.path.join(input_path, filename), sr=sample_rate)

        # compute maximum number of chunks fitting in source duration
        n = int( (len(source) - sr*overlap) / (duration*sr - overlap*sr))
        
        # if source file is shorter than duration, skip it
        if (n<=0):
            continue
        
        # Get one chunk of "duration" seconds at a time
        for i in range(n):
            nb_chunks += 1
            chunk_name = filename[0:-4] + '_chunk' + str(i).zfill(4)
            save_path = os.path.join(output_path, chunk_name + '.wav')
            # Save chunk, only if file doesn't already exist.
            if not os.path.exists(save_path):
                start_t = i*(duration-overlap)
                end_t = start_t + duration
                chunk = source[int(start_t*sr):int(end_t*sr)]
                librosa.output.write_wav(save_path, chunk, sr)

    printb('Done')

    return output_path, nb_files, nb_chunks


def filter_chunks(input_path, output_path, discard_label, duration=1, overlap=0, thresholds=[[0,0]], filenames=None):
    #Sanity check
    assert duration > 0, "duration must be strictly positive"
    assert overlap >= 0, "overlap must be positive or zero"
    assert overlap < duration, "overlap must be strictly less than duration"

    if not filenames:
        # Build audio filenames list (both wav and mp3)
        paths = glob.glob(os.path.join(input_path, '*.mp3'))
        paths.extend(glob.glob(os.path.join(input_path, '*.wav'))) 
        filenames = [os.path.basename(x) for x in paths]

    # Announce planned work
    printb("Processing "+str(len(filenames)) + " audio files.")

    # Walk the file list
    for filename in filenames:
        if (filename == '') or (filename.startswith('#')):
            continue

        print(filename)
        chunk_name = filename[0:-4] + '_chunk' + str(i).zfill(4)

        # Process annotations:
        for th in thresholds:
            label_file_name = Path(output_path, 'filter_U'+str(th[0])+'_G'+str(th[1]) + '.csv')
            label_file_exists = label_file_name.isfile()
            with open(label_file_name, 'a', newline='') as label_file:
                writer = csv.DictWriter(label_file, fieldnames=[
                            'name', 'start_t', 'end_t', 'strength'], delimiter=',')
                if not label_file_exists:
                    writer.writeheader()

                label_block_th = save_labels(filename, chunk_name, start_t, end_t, input_path, th)
                
                with open(os.path.join(annotations_path, annotation_filename), 'r') as f:
                    lines = f.read().split('\n')
                    sum_nobee = 0
                    for line in lines:
                        if (line == annotation_filename[0:-4]) or (line == '.') or (line == ''):
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
                        if label == discard_label and (tp1 - tp0) >= threshold:
                            sum_nobee = sum_nobee + min(tp1, chunk_end_t) - max(chunk_start_t, tp0)

                            if (sum_nobee > 0):
                                labels_th = ["nobee", round(sum_nobee/chunk_length, 3)]

                writer.writerow({'name': chunk_name, 'start_t': start_t, 'end_t': end_t,
                                'strength': label_block_th[1],  'label': label_block_th[0]})
    printb('---------------------- Done ----------------------')

    return

# states: state_labels=['active','missing queen','swarm' ]
def read_HiveState_fromSampleName(filename, states):
    label_state = 'other'
    for state in states:
        if state in filename.lower():
            # print("1 ", filename)
            label_state = state
    # incorporate condition for Nu-hive recordings which do not follow the same annotation: 'QueenBee' or 'NO_QueenBee'

    if label_state == 'other':
        if 'NO_QueenBee' in filename:
            # print("NO_QueenBee",label_state )
            label_state = states[1]
        else:
            label_state = states[0]
    return label_state


def write_Statelabels_from_beeNotBeelabels(path_save, path_labels_BeeNotBee, states=['active', 'missing queen', 'swarm']):

    # label_file_exists = os.path.isfile(path_save+'state_labels.csv')
    liste = []
    with open(path_labels_BeeNotBee, 'r') as rfile, \
            open(path_save+'state_labels.csv', 'w', newline='') as f_out:
        csvreader = csv.reader(rfile, delimiter=',')
        writer = csv.DictWriter(
            f_out, fieldnames=['sample_name', 'label'], delimiter=',')
        # if not label_file_exists:
        writer.writeheader()

        for row in csvreader:
            if not row[0] == 'sample_name':
                if row[4] == 'bee':
                    label_state = read_HiveState_fromSampleName(row[0], states)
                    # print(row[0],"label_state : ", label_state)
                    writer.writerow(
                        {'sample_name': row[0], 'label': label_state})
                else:
                    liste.append(row[0])
    return liste




def get_list_samples_names(path_audioSegments_folder, extension='.wav'):
    print(path_audioSegments_folder)
    sample_ids = [os.path.basename(x) for x in glob.glob(
        os.path.join(path_audioSegments_folder,'*'+extension))]
    return sample_ids
