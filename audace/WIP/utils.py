# Khellouf leila 
# utility functions
# python -m pip install SoundFile



#___________________________________________________________________________________________________
import glob
import os
#from info import i, printb, printr, printp, print
import glob
import os
#import librosa
import pdb
import csv
import json
import re
import numpy as np
import random
#import librosa.display
import IPython.display as ipd
from sklearn import preprocessing
from collections import Counter
from matplotlib import pyplot as plt
#from info import i, printb, printr, printp, print
#import muda
#import jams
from sklearn import svm
#import librosa
import keras
import scipy.io as sio
import io
from scipy.io import savemat
from scipy.sparse import csr_matrix
from os.path import dirname, join as pjoin
#_______________________________________________________________
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import plot_precision_recall_curve
#from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from keras.models import Sequential, Input, Model 
from keras.layers import Dense, Dropout, Flatten, Activation 
from keras.layers import Conv2D , MaxPooling2D
from keras.layers.normalization import BatchNormalization 
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, load_model
from keras.initializers import normal
from tensorflow.keras import layers
from tensorflow.keras import initializers
from sklearn.metrics import accuracy_score 
import pandas as pd 
import csv
import numpy as np
import scipy as sc
#________________________________________________________________
import timbre_descriptor as td
import scipy as sc
import my_tools as mt
from collections import namedtuple
import scipy
import scipy.signal
from scipy.io import wavfile

import matplotlib
import swipep as swp             # used for sing le-F0 estimation
import warnings                 # used for warning removal
import time               # used performance benchmark

nbits = 16;
MAX_VAL = pow(2,(nbits-1)) * 1.0;
class_names= ['missing queen', 'active' ]



#----------------------------------- parameters to change-----------------------------------#
block_size=1 # blocks of 1 second
thresholds=[0, 0.5]  # minimum length for nobee intervals: 0 or 5 seconds (creates one label file per threshold value)
path_audioFiles="D:\\datasets\\sounds\\BNB"+os.sep  # path to audio files
annotations_path="D:\\datasets\\sounds\\BNB"+os.sep # path to .lab files
path_save_audio_labels= 'D:\\datasets\\sounds\\BNB\\chunks_'+str(block_size)+'sec'+os.sep  # path where to save audio segments and labels files.
#----------------------------------- parameters to change-----------------------------------#
path_workingFolder='D:\\datasets\\sounds\\BNB\\chunks_'+str(block_size)+'sec'+os.sep  # path where to save audio segments and labels files.
labels2read= 'state_labels'
feature = 'MFCCs20'
path_working_MFCCs20= 'D:\\datasets\\sounds\\BNB\\chunks_'+str(block_size)+'sec'+'\\MFCCs20_matrix.mat'+os.sep
path_working_TTBox= 'D:\\datasets\\sounds\\BNB\\chunks_'+str(block_size)+'sec'+'\\TTBox_matrix.mat'+os.sep
path_working_stft= 'D:\\datasets\\sounds\\BNB\\chunks_'+str(block_size)+'sec'+'\\stft_matrix.mat'+os.sep
path_working_cqt= 'D:\\datasets\\sounds\\BNB\\chunks_'+str(block_size)+'sec'+'\\cqt_matrix.mat'+os.sep


path_save_audio_MFCCs= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\MFCCs20_matrix.mat'+os.sep 
path_save_audio_ttbox= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\ttb_mat'+os.sep 
path_save_audio_stft= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\stft_matrix.mat'+os.sep 
path_save_audio_cqt= 'C:\\Users\\PC\\python\\Stage\\dataset_BeeNoBee_2_second'+str(block_size)+'sec'+'\\cqt_matrix.mat'+os.sep 

nbits = 16
MAX_VAL = pow(2,(nbits-1)) * 1.0
target_names=['missing_queen', 'active']






#_____________________________________Calculate the features ____________________________________________________________________    


def raw_feature_fromSample( path_audio_sample, feature2extract ):
        
    s, Fs = librosa.core.load(path_audio_sample) # sr= 22050
    #m = re.match(r"\w+s(\d+)", feature2extract) #  m:  <re.Match object; span=(0, 7), match='MFCCs20'>
    #n_freqs=int(m.groups()[0]) ## n_freqs= 20
    #Melspec = librosa.feature.melspectrogram(audio_sample, n_mels = n_freqs) # computes mel spectrograms from audio sample, 
    
    if 'MFCCs' in feature2extract:
        n_freqs = int(feature2extract[5:len(feature2extract)]) ## n_freqs==20
        Melspec = librosa.feature.melspectrogram(s, sr=Fs)
        x = librosa.feature.mfcc(S=librosa.power_to_db(Melspec),sr=Fs, n_mfcc = n_freqs)
        #x = librosa.feature.mfcc(s ,Fs=Fs, n_mfcc = n_freqs)
        #print(x.shape)
    # Short-time Fourier transform    
    elif 'stft' in feature2extract:
        x= np.abs(librosa.stft(s , n_fft=2048, hop_length=514, win_length=2048, window='hann', center=True, dtype=np.complex64, pad_mode='reflect'))
        
    # constante Q transform  Using a higher frequency resolution  
    elif 'cqt' in  feature2extract:
        x = np.abs(librosa.cqt(s, sr=Fs, fmin=librosa.note_to_hz('C2'), n_bins=60 * 2, bins_per_octave=12 * 2))
        
    # timber tool box     
    elif 'TTBOX' in feature2extract:
        r_Fs = librosa.resample(s, Fs, 44100)
        r_Fs=44100
        s= s/MAX_VAL;
        N = len(s);
       
        desc_TEE, desc_AS, desc_STFTmag, desc_STFTpow, desc_Har, desc_ERB, desc_GAM =td.compute_all_descriptor(s, r_Fs);
        desc  = [desc_TEE, desc_AS, desc_STFTmag, desc_STFTpow, desc_Har, desc_ERB, desc_GAM];

        ### Time serie integration
        param_val, field_name = td.temporalmodeling(desc);
        x=param_val
    
        
    else:
        x = Melspec

    return x 



def get_features_from_samples(path_audio_samples, sample_ids, raw_feature, normalization, high_level_features ): 
    #normalization = NO, z_norm, min_max
    ## function to extract features 
    #high_level_features = 0 or 1 
    #file_path = os.path.isfile(path_save_audio_labels+'matrix.mat'+'.csv') 
    n_samples_set = len(sample_ids) # 4
    feature_Maps = []
    if raw_feature== 'MFCCs20': 
         path_working= path_working_MFCC20
    elif  raw_feature=='TTBOX': 
         path_working= path_working_TTBox
    elif raw_feature=='stft':
         path_working= path_working_stft
    else: 
        path_working= path_working_cqt
    
    for sample in sample_ids:
        # raw feature extraction:
        print("sample: ",sample)
        x = raw_feature_fromSample( path_audio_samples+sample, raw_feature ) # x.shape: (4, 20, 2584)
       # print("x.shape: ",x.shape)
       # Sauvgarder les x dans un fichier .mat 
     
        b = csr_matrix(x)
        savemat(path_working+ sample[:-4] + '.mat', {'b': b})
         
                         
       
        if not normalization == 'NO':
             x_norm = featureMap_normalization_block_level(x, normalizationType = normalization) 
        else: x_norm = x
        
        if high_level_features:
            # high level feature extraction:
            if 'MFCCs' in raw_feature:
                X = compute_statistics_overMFCCs(x_norm, 'yes') # X.shape: (4 , 120)
            else: 
                X = compute_statistics_overSpectogram(x_norm)
                
            feature_map=X
        else:
            feature_map=x_norm
        
        
        feature_Maps.append(feature_map)
        
    return feature_Maps

def labels2binary(pos_label, list_labels):  # pos_label = missing queen / nobee
    list_binary_labels=[]
    for l in list_labels:
        #print("l=", l, "pos_label= ", pos_label)
        if l == pos_label:
            list_binary_labels.append(1)
        else:
            list_binary_labels.append(0)
    return list_binary_labels

def get_GT_labels_fromFiles(path_labels, sample_ids, labels2read) : #labels2read =  name of the label file    
    
    ##reads labels files and retreives labels for a sample set given by sample_ids
    # input:  path_labels: where label file is 
    #         sample_ids: list of sample names that we want the label
    #         labels2read: name of the labels file: state_labels, labels_BeeNotBee_th0 ...
    
    # output: list of string labels, in same order as sample_ids list
    
    labels = []
    fileAsdict={}
    
    with open(path_labels + labels2read+'.csv', 'r') as labfile:
        csvreader = csv.reader(labfile, delimiter=',')    
        for row in csvreader:
            if not row[0] == 'sample_name':
               # print("row[0]",row[0]) # CF001 - Missing Queen - Day -__segment0
               # print("row[-1]:", row[-1])  # bee or nobee
                fileAsdict[row[0]]=row[-1]   # row[-1] = '/missing queen/active' or 'bee/nobee'
            
    for sample in sample_ids:
        # print(sample)
        #print(fileAsdict[sample[0:-4]]) # bee nobee or unknown 
        labels.append(fileAsdict[sample[0:-4]])  #remove .wav extension
    
       
    return labels  

#----------------------------------------------------data augmentation-------------------------------------------------
def raw_feature_fromSample_withAUGMENTATION(  path_audio_sample,sample_name, list_features ,augmentingFactor):
    
    print("path_audio_sample + sample_name= ",path_audio_sample + sample_name)
    audio_sample_or, sr = librosa.core.load(path_audio_sample + sample_name)
    # list_audio_samples = [audio_sample_or]
    spectrogram_original =[]
    
    dict_augmented_wav_feature = {}
    
    # Applying a deformation
    # pitch = muda.deformers.LinearPitchShift(n_samples=5, lower=-1, upper=1)
    # in this example we have simple linear pitch shift deformer to generate five perturbations of an input
    # 
    pitch = muda.deformers.LinearPitchShift(n_samples=augmentingFactor, lower=-1, upper=1) 
    jam=jams.JAMS()
    j_orig = muda.jam_pack(jam, _audio=dict(y=audio_sample_or, sr=sr))
    # each deformed example is saved to disk
    for i, jam_out in enumerate(pitch.transform(j_orig)):
        y = jam_out.sandbox.muda._audio['y']
        sr = jam_out.sandbox.muda._audio['sr']

        #list_audio_samples.append(y)  #list with original_audio plus every modified version of the same audio
        dict_augmented_wav_feature[sample_name[0:-4]+'_AG'+str(i)]=[y]
                
    # Extract features for the set of samples: original and  augmented...
    for feature2extract in list_features:
        m = re.match(r"\w+s(\d+)", feature2extract)
        n_freqs=int(m.groups()[0])
        
        
        # EXTRACT FOR ORIGINAL SAMPLE:
        Melspec = librosa.feature.melspectrogram(audio_sample_or, n_mels = n_freqs) # computes mel spectrograms from audio sample, 
        
        if 'LOG' in feature2extract: #'LOG_MELfrequencies48'
            Melspec=librosa.feature.melspectrogram(audio_sample_or, sr=sr, n_mels=n_freqs)
            x=librosa.power_to_db(Melspec+1)
            
        elif 'MFCCs' in feature2extract:
            n_freqs = int(feature2extract[5:len(feature2extract)])
            Melspec = librosa.feature.melspectrogram(audio_sample_or, sr=sr)
            x = librosa.feature.mfcc(S=librosa.power_to_db(Melspec),sr=sr, n_mfcc = n_freqs)
            
        else:
            x = Melspec
        spectrogram_original.append(x)   # append each feature!     
        
        
        # EXTRACT FOR SET OF AUGMENTED SAMPLES:
                
        for aug_audio_sample_name in dict_augmented_wav_feature.keys():
        
        
            Melspec = librosa.feature.melspectrogram(dict_augmented_wav_feature[aug_audio_sample_name][0], n_mels = n_freqs) # computes mel spectrograms from audio sample, 
        
            if 'LOG' in feature2extract: #'LOG_MELfrequencies48'
                Melspec=librosa.feature.melspectrogram(dict_augmented_wav_feature[aug_audio_sample_name][0], sr=sr, n_mels=n_freqs)
                x=librosa.power_to_db(Melspec+1)
                
            elif 'MFCCs' in feature2extract:
                n_freqs = int(feature2extract[5:len(feature2extract)])
                Melspec = librosa.feature.melspectrogram(dict_augmented_wav_feature[aug_audio_sample_name][0], sr=sr)
                x = librosa.feature.mfcc(S=librosa.power_to_db(Melspec),sr=sr, n_mfcc = n_freqs)
                
            else:
                x = Melspec
        
            dict_augmented_wav_feature[aug_audio_sample_name].append(x)    #dict; {sample_name_AG1 : [y,spectrogram], sample_name_AG2 : [y,spectrogram]}
            
    
    
    return dict_augmented_wav_feature, spectrogram_original    
 
    
    
    
def get_features_from_samples_withAUGMENTATION(path_audio_samples, sample_ids, raw_feature, normalization, high_level_features ,augmentingfactor):
    
     #normalization = NO, z_norm, min_max
    # ## function to extract features 
    # #high_level_features = 0 or 1 
    # #augmentingfactor = 0 won't do any augmentation
    print(path_audio_samples)
    n_samples_set = len(sample_ids) # 4
    feature_Maps = []

    for sample in sample_ids:
         
        # # raw feature extraction:
        print("sample= ",path_audio_samples+ sample)
        x, list_x_augm = raw_feature_fromSample_withAUGMENTATION( path_audio_samples, sample, raw_feature , augmentingfactor ) # x.shape: (4, 20, 2584)

    # #for x in list_x :

        # #normalization here:
        if not normalization == 'NO':
             x_norm = featureMap_normalization_block_level(x, normalizationType = normalization) 
        else: x_norm = x

        if high_level_features:
            # # high level feature extraction:
            if 'MFCCs' in raw_feature:
                X = compute_statistics_overMFCCs(x_norm, 'yes') # X.shape: (4 , 120)
            else: 
                X = compute_statistics_overSpectogram(x_norm)

            feature_map=X
        else:
             feature_map=x_norm


        feature_Maps_original.append(feature_map)

        feature_maps_augmented=[]  

    return feature_Maps
    


#__________________________________________4 folds _Cross Validation ______________________________________________________________
# return the 4 folds : fold1= CF001+ CF003, fold2= CJ001+GH001, .....
def get_list_samples_name_( mtx , path_audioSegments, extension='.mat' ):
    states=['active','missing queen','swarm' ]
    labels1=[]
    labels2=[]
    labels3=[]
    labels4=[]
    Y1=[]
    Y2=[]
    Y3=[]
    Y4=[]
    sample_ids1=[]
    sample_ids2=[]
    sample_ids3=[]
    sample_ids4=[]
    ruche1=[]
    ruche2=[]
    ruche3=[]
    ruche4=[]
    
    for x in glob.glob(path_audioSegments+'*'+extension): 
        size=len(path_audioSegments)
        sample=x[size:]
        #print(sample)
        if sample[0:5]=="CF001" or sample[0:5]=="CF003":
            
            sample_ids1.append(sample)
            l= read_HiveState_fromSampleName( sample, states)
            labels1.append(l)
            m=scipy.io.loadmat(x)
            ruche1.append(m[mtx])
        elif sample[0:5]=="CJ001" or sample[0:5]=="GH001" :
            
            sample_ids2.append(sample)
            l= read_HiveState_fromSampleName( sample, states)
            labels2.append(l)
            m=scipy.io.loadmat(x)
            ruche2.append(m[mtx])
        elif sample[0:5]=="Hive1": 
            
            sample_ids3.append(sample)
            l= read_HiveState_fromSampleName( sample, states)
            labels3.append(l)
            m=scipy.io.loadmat(x)
            ruche3.append(m[mtx])
        else : 
            sample_ids4.append(sample)
            l= read_HiveState_fromSampleName( sample, states)
            labels4.append(l)
            m=scipy.io.loadmat(x)
            ruche4.append(m[mtx])
    
    Y1= labels2binary('active', labels1)
    Y2= labels2binary('active', labels2)
    Y3= labels2binary('active', labels3)
    Y4= labels2binary('active', labels4)
    return ruche1,Y1,labels1, sample_ids1, ruche2,Y2,labels2, sample_ids2, ruche3,Y3,labels3, sample_ids3, ruche4,Y4,labels4, sample_ids4
#ruche1,Y1,labels1, sample_ids1, ruche2,Y2,labels2, sample_ids2, ruche3,Y3,labels3, sample_ids3, ruche4,Y4,labels4, sample_ids4=get_list_samples_name_MFCCS('b', path_save_audio_MFCCs)

# Define the 4 Fold cross validation 
def cross_validation_4folds(fold, ruche1,Y1, ruche2,Y2, ruche3,Y3, ruche4,Y4, sample_ids1 , sample_ids2 , sample_ids3 , sample_ids4):
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]
    sample_ids_train=[]
    sample_ids_test=[]
    if fold==1:
        print("fold==1")
        x_train=ruche2+ ruche3+ ruche4
        y_train=Y2+ Y3+ Y4
        x_test=ruche1
        y_test=Y1
        sample_ids_train=sample_ids2+sample_ids3+sample_ids4
        sample_ids_test=sample_ids1
    elif fold==2:
        print("fold==2")
        x_train=ruche1 + ruche3+ ruche4
        y_train=Y1+ Y3+ Y4
        x_test=ruche2
        y_test=Y2
        sample_ids_train=sample_ids1+sample_ids3+sample_ids4
        sample_ids_test=sample_ids2
    elif fold==3:
        print("fold==3")
        x_train=ruche1 + ruche2+ ruche4
        y_train=Y1+ Y2+ Y4
        x_test=ruche3
        y_test=Y3
        sample_ids_train=sample_ids1+sample_ids2+sample_ids4
        sample_ids_test=sample_ids3
    else:
        print("fold==4")
        x_train=ruche1 + ruche2+ ruche3
        y_train=Y1+ Y2+ Y3
        x_test=ruche4
        y_test=Y4
        sample_ids_train=sample_ids1+sample_ids2+sample_ids3
        sample_ids_test=sample_ids4
        
    return  x_train, x_test, y_train, y_test, sample_ids_train, sample_ids_test    

        
#x_train, x_test, y_train, y_test, sample_ids_train, sample_ids_test= cross_validation_4folds(3, ruche1,Y1, ruche2,Y2, ruche3,Y3, ruche4,Y4, sample_ids1 , sample_ids2 , sample_ids3 , sample_ids4)  
#len(x_train), len(x_test), len(y_train), len(y_test), len(sample_ids_train)

# Fit and evaluate the model 

def train_and_evaluate_model(model, X_train, Y_train, X_test, Y_test, y_test ,  nb_epoch, batch_size ,model_filename):
    # y_test is labels, Y_test is categorical labels
    
    print('Train...')
    target_names=['missing_queen', 'active']
    stopping = EarlyStopping(monitor='val_accuracy', patience=50)
    checkpointer = ModelCheckpoint(filepath=model_filename, verbose=1, save_best_only=True ,save_weights_only=False, monitor='val_accuracy' , mode='max')
    results= model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[stopping, checkpointer], verbose=2 , validation_data=(X_test, Y_test))
    # prediction
    best_model = load_model(model_filename)
    probabilities = best_model.predict(X_test, batch_size=batch_size)
    predictions = probabilities.argmax(axis=-1)
    best_acc = accuracy_score(y_test, predictions)
    print('Accuracy score (best model): {}'.format(best_acc))
    #print("classification report: ", classification_report(y_test, predictions))
    report = classification_report(y_test, predictions , target_names=target_names, output_dict=True)
    cnf_matrix = confusion_matrix(y_test, predictions )
    return results, best_acc, report, cnf_matrix 


# train_test_split the data we take 0.7% training and 0.3% for the test 

def constrainedsplit(y_train, x_test, y_test, ratio=0.7):
	
	nb_class = len(np.unique(y_test));   ## nb de classes
	
	N_max_test = round(len(y_train)/ratio * (1-ratio))
	
	N_test  = np.zeros(nb_class);
	N_train = np.zeros(nb_class);
	for i in range(nb_class):
		N_test[i]  = round(len(y_test[y_test==i]))
		N_train[i] = round(len(y_train[y_train==i]))
	
	I = np.argsort(N_test);
	for i in range(nb_class):
		j = I[i]
		
		N_max_tmp = round(N_train[j] / ratio * (1-ratio));
		
		n_tmp = int(min( N_test[j], N_max_tmp))
		x_tmp = x_test[y_test==j,:]
		y_tmp = y_test[y_test==j]
		if i == 0:
			x_test_out = x_tmp[0:n_tmp, :]
			y_test_out = y_tmp[0:n_tmp]
		else:
			x_test_out = np.concatenate([x_test_out, x_tmp[0:n_tmp, :]])
			y_test_out = np.concatenate([y_test_out, y_tmp[0:n_tmp]])
	
	return x_test_out,y_test_out




#_____________________________________________________  SVM functions_________________________________________________________



# SVM CLASSIFICATION:
    

def SVM_Classification_BeehiveSTATE(X_flat_train, y_train, X_flat_test, y_test, kerneloption='rbf'):

    print('\n')
    printb('Starting classification with SVM:')
    Test_Preds=[]
    Train_Preds=[]
    Test_Preds_Proba=[]
    Train_Preds_Proba=[]
    Test_GroundT=[]
    Train_GroundT=[]
   
    print('\n')
    printb('classification Beehive State into : Active or Missing Queen')
        
    #train :
    CLF = svm.SVC(kernel=kerneloption, probability=True)
    CLF.fit(X_flat_train, y_train)
    y_pred_train = CLF.predict(X_flat_train)
    y_pred_proba_train = CLF.predict_proba(X_flat_train)
    
    Train_GroundT = y_train
    Train_Preds = y_pred_train
    Train_Preds_Proba = y_pred_proba_train[:,1]
    
    # test:
    y_pred_test = CLF.predict(X_flat_test)
    y_pred_proba_test = CLF.predict_proba(X_flat_test)
    Test_GroundT= y_test
    Test_Preds = y_pred_test
    Test_Preds_Proba = y_pred_proba_test[:,1]

    return CLF, Test_GroundT, Train_GroundT, Test_Preds, Train_Preds, Test_Preds_Proba, Train_Preds_Proba 

#_____________________________________________________CNN Classification __________________________________________

def deep_model(size):
    # Neural Network Architecture 
    model=Sequential()
    # size= ( 20,44, 1)
   # for i in range(1, num_cnn_layers+1):
         
    model.add(Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=size , padding='same', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    model.add(BatchNormalization())
   # model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    # we add the dropout to skip the overfitting 
    # model.add(Dropout(0.25))
    # add the batch_normalization



    model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
   # model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))

    # we add the dropout to skip the overfitting 
    # model.add(Dropout(0.25))

    model.add(Conv2D(16, kernel_size=(3,1), activation='relu', padding='same',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    
    #model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    #model.add(Dropout(0.25))

    model.add(Conv2D(16, kernel_size=(3,1), activation='relu', padding='same',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))

    # we add the dropout to skip the overfitting 
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(32 , activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(2, activation='softmax',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model









#______________________________________TTBOX+DenseNet_____________________________________________________________________
def Dense_Net(size): 
    # size=(164,1 )
    # Create the model
    model=Sequential()
    # the first Dense layer
    model.add(Flatten(input_shape=size))
    model.add(Dense(328 ,  activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    #model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # the second dense layer 
    model.add(Dense(328 , activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    #model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # the third dense layer 
    model.add(Dense(328,  activation='relu',  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    #model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # the output layer 
    model.add(Dense(2, activation="softmax" ,  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model
















#_______________________________________ For Plotting_________________________________________________________________________________
            
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=25)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_accuracy_val_accuracy(model_history):

# Plot to see how models are performing 

    plt.title('Accuracies vs Epochs ')
    plt.plot(model_history[0].history['accuracy'], label='Training Fold 1')
    plt.plot(model_history[1].history['accuracy'], label='Training Fold 2')
    plt.plot(model_history[2].history['accuracy'], label='Training Fold 3')
    plt.plot(model_history[3].history['accuracy'], label='Training Fold 4')
    plt.legend()
    plt.show()
    plt.title('Train Accuracy vs Val Accuracy')
    plt.plot(model_history[0].history['accuracy'], label='Train Accuracy Fold 1', color='black')
    plt.plot(model_history[0].history['val_accuracy'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")
    plt.plot(model_history[1].history['accuracy'], label='Train Accuracy Fold 2', color='red', )
    plt.plot(model_history[1].history['val_accuracy'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")
    plt.plot(model_history[2].history['accuracy'], label='Train Accuracy Fold 3', color='green', )
    plt.plot(model_history[2].history['val_accuracy'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
    plt.plot(model_history[3].history['accuracy'], label='Train Accuracy Fold 4', color='steelblue', )
    plt.plot(model_history[3].history['val_accuracy'], label='Val Accuracy Fold 4', color='steelblue', linestyle = "dashdot")



    plt.legend()
    plt.show()
    
def save_confusion_matrix(cnf_matrix, filename, class_names):
    fig = plt.figure()
    # Plot non-normalized confusion matrix

    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=filename)
    plt.savefig('confusion_matrix'+str(filename)+'.jpg')
    
    
    
    
    
    
    
    
#_____________________________________________________________________________________________________________




def get_list_samples_name_MFCC(path_audioSegments, extension='.mat'):
    states=['active','missing queen','swarm' ]
    X_ttbox=[]
    labels=[]
    Y=[]
    sample_ids=[]
    
    for x in glob.glob(path_audioSegments+'*'+extension): 
        size=len(path_audioSegments)
        sample=x[size:]
        sample_ids.append(sample)
        l= read_HiveState_fromSampleName( sample, states)
        labels.append(l)
        m=scipy.io.loadmat(x)
        X_ttbox.append(m['b'])
    
    Y= labels2binary('active', labels)
        
    return X_ttbox,  labels , Y, sample_ids
def get_list_samples_name_TTBOX(path_audioSegments, extension='.mat'):
    states=['active','missing queen','swarm' ]
    X_ttbox=[]
    labels=[]
    Y=[]
    sample_ids=[]
  
    for x in glob.glob(path_audioSegments+'*'+extension): 
        size=len(path_audioSegments)
        sample=x[size:]
        sample_ids.append(sample)
        l= read_HiveState_fromSampleName( sample, states)
        labels.append(l)
        m=scipy.io.loadmat(x)
        X_ttbox.append(m['ttb_vec'])
    
    Y= labels2binary('active', labels)
        
    return X_ttbox,  labels , Y, sample_ids



#_________________________________________________________________





#_____________________________________TTBOX+SVM________________________________________________________________________

def get_list_samples_name(path_audioSegments, extension='.mat'):
    X_ttbox=[]
    sample_ids=[]
    list_ttbox=[os.path.basename(x) for x in glob.glob(path_audioSegments+'*'+extension)]
    for x in glob.glob(path_audioSegments+'*'+extension):
        
        sample_ids.append(x[63:])
        m=scipy.io.loadmat(x)
        X_ttbox.append(m['ttb_vec'])
    return X_ttbox, sample_ids, list_ttbox





def fit_and_evaluate(train_x, val_x, train_y, val_y, EPOCHS=50, BATCH_SIZE=145 ):
    model=None
    model=deep_model(( 20,44, 1))
    results= model.fit(train_x, train_y, epochs=EPOCHS, batch_size= BATCH_SIZE, callbacks=[early_stopping, model_checkpoint], verbose=1, validation_split=0.1)
    print("Val Score :", model.evaluate(val_x, val_y))
    return results 

def BalanceData_online(y_set, x_set, sample_ids_set):
    
    ## balances already processed data (X and Y, just before classifier) by replicating samples of the least represented class.
    # input: y_set - binary labels of set, x_set - feature_maps of set, sample_ids_set - sample names in set, ( all have the same order!)
    # output: X, Y and sample_ids with replicated samples concatenated 
    
 
    printb( 'Balancing training data:' )
    print('will randomly replicate samples from least represented class')
    
    x2concatenate = x_set
    y2concatenate = y_set
    sample_ids2concatenate = sample_ids_set
    
    dict_items_replicate = get_items2replicate(y_set,sample_ids_set )
    #print("dict_items_replicate: ",dict_items_replicate)
    
    for i in range(len(sample_ids_set)):
        if sample_ids_set[i] in dict_items_replicate.keys() :
            
            sample_ids2concatenate =np.concatenate([sample_ids2concatenate, [sample_ids_set[i]]*dict_items_replicate[sample_ids_set[i]]])
            y2concatenate = np.concatenate([y2concatenate, [y_set[i]]*dict_items_replicate[sample_ids_set[i]]])
            x2concatenate = np.concatenate([x2concatenate, [x_set[i]]*dict_items_replicate[sample_ids_set[i]]])
            
    return y2concatenate, x2concatenate, sample_ids2concatenate


def get_items2replicate(list_Binary_labels, list_sample_ids):
    
    # get the samples to be replicated.
    # input: list of labels and sample_ids with same oreder!
    # ouptut: dictionary keys:name of samples to be replicated,  value: Number of times to replicate that sample.
    
    #assert( len(list_Binary_labels) - len(list_sample_ids) == 0), ('arguments should have the same number of elements)
    dict_items_replicate={}
    
    n_samples = len(list_Binary_labels)# 193
    #print("n_samples: ", list_Binary_labels)
    n_positive_labels = sum(list_Binary_labels)#158 = le nbr de 1
    #print("n_positive_labels: ",n_positive_labels)
    n_negative_labels = n_samples - n_positive_labels #35= le nbr de 0
    #print("n_negative_labels: ",n_negative_labels)
    
    pos_samples=[]
    neg_samples=[]
    
    for i in range(n_samples):
        if list_Binary_labels[i] == 1 :
            #print("list_sample_ids[i]= : ", list_sample_ids[i])
            pos_samples.append(list_sample_ids[i])
        else: 
            neg_samples.append(list_sample_ids[i])
            
    if n_positive_labels > n_negative_labels:
        #print(n_positive_labels, n_negative_labels)
        # Replicate negative samples as needed:
        dif=n_positive_labels-n_negative_labels
        items_replicate=random.choices(neg_samples, k=dif)
       # print("neg_samples= ",neg_samples, "items_replicate=",items_replicate)
 
    elif n_positive_labels < n_negative_labels:
        dif=n_negative_labels-n_positive_labels
        items_replicate=random.choices(pos_samples, k=dif)
              
    dict_items_replicate=Counter(items_replicate)
    #print(dict_items_replicate)
    return dict_items_replicate
