#===== Standard imports
import os
import glob
import csv

#===== 3rd party imports
import librosa
import numpy as np

#===== Local imports
from info import i, printb, printr, printp, print




def read_beeNotBee_annotations_saves_labels(audiofilename, block_name,  blockStart, blockfinish, annotations_path, threshold=0):
    
    
    ## function: reads corresponding annotation file (.lab) and assigns a label to one block/sample. Appends label into csv file.
    ##
    ## inputs: 
    ## audiofilename = name of the audio file (no path), block_name = name of the sample/segment,  blockStart = time point in seconds where block starts, blockfinish = time point in seconds where block ends, annotations_path = path to annotations folder (where .lab files are), threshold = value tor threshold. 
    ##
    ## outputs:
    ## label_th= 2 element list, [0] = a label (bee / nobee) for the block and threshold considered; [1] = label strength, value that reflects the proportion of nobee interval in respect to the whole block.
    # threshold gives the minimum duration of the no bee intervals we want to consider.
    # threshold=0 uses every event as notBee whatever the duration
    # threshold=0.5 disregards intervals with less than half a second duration.
    
    block_length=blockfinish-blockStart
    
    if audiofilename.startswith('#'):
        annotation_filename=audiofilename[1:-4]+'.lab'
    else :
        annotation_filename=audiofilename[0:-4]+'.lab'
        
        
    try:    
        with open(annotations_path + os.sep + annotation_filename,'r') as f:
            # EXAMPLE FILE:
            
            # CF003 - Active - Day - (223)
            # 0	8.0	bee
            # 8.01	15.05	nobee
            # 15.06	300.0	bee 
            # .
            #
            
            # all files end with a dot followed by an empty line.

            #print(annotations_path + os.sep + annotation_filename)
            lines = f.read().split('\n')
        
            labels_th=['bee', 0.0]
            label2assign='bee'
            label_strength=0
            intersected_s = 0
            intersected_s2 = 0
            count = 0
            for line in lines:
                if (line == annotation_filename[0:-4]) or (line == '.') or (line ==''):
                    #ignores title, '.', or empty line on the file.
                    continue
                
                count = count + 1
                #print(count,": ",line)
                parsed_line= line.split('\t')    
                
                assert (len(parsed_line)==3), ('expected 3 fields in each line, got: '+str(len(parsed_line))) 
                
                
                tp0=float(parsed_line[0])
                tp1=float(parsed_line[1])
                annotation_label=parsed_line[2]
                if blockfinish < tp0: # no need to read further nobee intervals since annotation line is already after block finishes
                    break
                    
                if annotation_label== 'nobee':
                    if tp1-tp0 >= threshold:  # only progress if nobee interval is longer than defined threshold.
                        if tp0 > blockStart and tp0 <= blockfinish and tp1 >= blockfinish:
                            intersected_s=intersected_s + (blockfinish-tp0)    
                            # |____________########|########
                            # bs          tp0      bf      tp1 
                        elif tp1 >= blockStart and tp1 < blockfinish and tp0 <= blockStart:
                            intersected_s=intersected_s+ (tp1-blockStart)
                            # #####|########_____|
                            # tp0  bs     tp1    bf
                        elif tp1 >= blockStart and tp1 <= blockfinish and tp0 >= blockStart and tp0 <= blockfinish:
                            intersected_s=intersected_s+ (tp1-tp0)
                            # |_____########_____|
                            # bs   tp0    tp1    bf
                        elif tp0 <= blockStart and tp1 >= blockfinish:
                            intersected_s=intersected_s + (blockfinish-blockStart)
                            #  ####|############|####
                            # tp0  bs           bf  tp1



                    if intersected_s > 0:
                        intersected_s2 = intersected_s2 + min(tp1, blockfinish) - max(blockStart, tp0)
                        if (intersected_s != intersected_s2):
                            print("What the fuck?")
                        label2assign = 'nobee'
                        
                    label_strength= intersected_s/block_length # proportion of nobee length in the block
                    labels_th= [label2assign, round(label_strength,3)]  # if label_strength ==0 --> bee segment 
                    
                    
            #assert (blockfinish <=tp1 ), ('the end of the request block falls outside the file: block ending: '+ str(blockfinish)+' end of file at: '+ str(tp1))
            
                
    except FileNotFoundError as e:
        print(e, '--Annotation file does not exist! label as unknown')
        #print(annotation_filename=audiofilename[0:-4]+'.lab')
        label2assign = 'unknown'
        label_strength=-1
        labels_th = [label2assign, label_strength]
            
    # except Exception as e1:
        # print('unknown exception: '+str(e1))
        #quit
    
    return labels_th


def build_chunks( input_path, output_path, duration , sample_rate, thresholds, use_annotations = True, save_chunks= True):
    # Build audio filenames list (both wav and mp3)
    filenames = [os.path.basename(x) for x in glob.glob(input_path+'*.mp3')]
    filenames.extend([os.path.basename(x) for x in glob.glob(input_path+'*.wav')])
    
    # Create output directory, if needed
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Announce planned work
    printb("Processing "+str(len(filenames)) + " audio files.")
    #print(filenames)
    
    for filename in filenames:
        offset=0
        chunk_id =0
        print(filename)
        while 1:
            try:
                ## Read one chunk of "duration" seconds at a time
                chunk, sr = librosa.core.load(input_path + filename, sr=sample_rate, offset=offset, duration=duration)
            except ValueError as e:
                e
                if 'Input signal length' in str(e):
                    chunk=np.arange(0)
            except FileNotFoundError as e1:
                print(e1, ' but continuing anyway')
                
            if chunk.shape[0] > 0:    #when total length = multiple of blocksize, results that last block is 0-lenght, this if bypasses those cases.
                
                chunk_name = filename[0:-4] + '_chunk' + str(chunk_id).zfill(4);
                
                # Process annotations, if requested:
                if use_annotations:
                    start_t = offset
                    end_t   = offset + duration
                    
                    for th in thresholds:
                        label_file_exists = os.path.isfile(output_path+'labels_th'+str(th)+'.csv')
                        with open(output_path + 'labels_th' + str(th)+'.csv','a', newline='') as label_file:
                            writer =csv.DictWriter(label_file, fieldnames=['name', 'start_t','end_t', 'strength', 'label'], delimiter=',')
                            if not label_file_exists:
                                writer.writeheader()
                            ##  print("start read_beeNotBee_annotation_saves_labels")
                            label_block_th=read_beeNotBee_annotations_saves_labels(filename, chunk_name,  start_t, end_t, input_path, th)                            
                            # print("label_block_th : ", label_block_th)                           
                            writer.writerow({'name': chunk_name, 'start_t': start_t, 'end_t': end_t , 'strength': label_block_th[1],  'label': label_block_th[0]} )
                            # print('-----------------Wrote label for th '+ str(th)+' seconds of segment'+str(block_id)  ) 
                    

                # MAKE BLOCK OF THE SAME SIZE:
                if chunk.shape[0] < duration*sr:   
                    chunk = uniform_block_size(chunk, duration*sr, 'repeat')
                    print('-----------------Uniformizing block length of segment'+str(chunk_id)  ) 

                        
            
                # Save chunk, if requested:
                if save_chunks and (not os.path.exists(output_path+chunk_name+'.wav')): #saves only if option is chosen and if block file doesn't already exist.
                    librosa.output.write_wav(output_path + chunk_name+'.wav', chunk, sr)
                    #print( '-----------------Saved wav file for segment '+str(block_id))
                    
            else :
                #print('----------------- no more segments for this file--------------------------------------')
                # print('\n')
                break
            offset += duration
            chunk_id += 1
    printb('______________________________No more audioFiles___________________________________________________')

    return 


def uniform_block_size(undersized_block, block_size_samples, method='repeat' ):

    lengthTofill=(block_size_samples)-(undersized_block.size)
    if method == 'zero_padding':
        new_block=np.pad(undersized_block, (0,lengthTofill), 'constant', constant_values=(0) )

    elif method=='mean_padding':
        new_block=np.pad(undersized_block, (0,lengthTofill), 'mean' )
    
    elif method=='repeat':        
        new_block= np.pad(undersized_block, (0,lengthTofill), 'reflect')
    else:
        print('methods to choose are: \'zero_padding\' ,\'mean_padding\' and \'repeat\' ' )
        new_block=0

    return new_block


def read_HiveState_fromSampleName( filename, states):   #states: state_labels=['active','missing queen','swarm' ]
    label_state='other'
    for state in states:
        if state in filename.lower():
            # print("1 ", filename)
            label_state = state
    #incorporate condition for Nu-hive recordings which do not follow the same annotation: 'QueenBee' or 'NO_QueenBee'
    
    if label_state=='other':
        if 'NO_QueenBee' in filename:
            ##print("NO_QueenBee",label_state )
            label_state = states[1]
        else:
            label_state=states[0]
    return label_state


def write_Statelabels_from_beeNotBeelabels(path_save, path_labels_BeeNotBee, states=['active','missing queen','swarm' ]):
    
    #label_file_exists = os.path.isfile(path_save+'state_labels.csv')
    liste=[]
    with open(path_labels_BeeNotBee, 'r' ) as rfile, \
    open(path_save+'state_labels.csv', 'w', newline='') as f_out:
        csvreader = csv.reader(rfile, delimiter=',')
        writer= csv.DictWriter(f_out, fieldnames=['sample_name', 'label'], delimiter=',') 
        #if not label_file_exists:
        writer.writeheader()
        
        for row in csvreader:
            if not row[0]=='sample_name':
                if row[4]=='bee':
                    label_state=read_HiveState_fromSampleName(row[0], states)
                    #print(row[0],"label_state : ", label_state)
                    writer.writerow({'sample_name':row[0], 'label':label_state})
                else:   liste.append(row[0])  
    return liste


def get_list_samples_names(path_audioSegments_folder, extension='.wav'):
    sample_ids=[os.path.basename(x) for x in glob.glob(path_audioSegments_folder+'*'+extension)]
    return sample_ids