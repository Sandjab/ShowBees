# Experiments

| Id | Mnemo | Dataset| Description |
|---|---|---|---|
|**EXP001**|MFCC+DNN| BNB | Blaaaaaaah Bmahhhhh Béééééé|



## General pre-processing of audio files
In order to produce samples usable for classification, hive recording audio files are segmented into chunks:
- of a specific duration (DUR), 
- resampled at a specific sample rate (SR), 
- and a thresold (THR) is applied in order to retain only files with less than HTS seconds of external noise.




# Datasets
## BNB data sets
The **BNB** dataset is a subset of the reference "to bee or not to bee' annotated dataset used in [1] (corresponding sound files and documentation is available for download at https://zenodo.org/record/1321278 )

This subset consists in the following 48 audio files, both mp3 and wav, of various durations, for a total of xx hours of recording on 6 different hives, over different days and time of the day

| File Name  (without extension)|
| --- |
| `CF001 - Missing Queen - Day -` |
| `CF003 - Active -   Day - (214)` |
| `CF003 - Active - Day - (215)` |
| `CF003 - Active -   Day - (216)` |
| `CF003 - Active - Day - (217)` |
| `CF003 - Active -   Day - (218)` |
| `CF003 - Active - Day - (219)` |
| `CF003 - Active -   Day - (220)` |
| `CF003 - Active - Day - (221)` |
| `CF003 - Active -   Day - (222)` |
| `CF003 - Active - Day - (223)` |
| `CF003 - Active -   Day - (224)` |
| `CF003 - Active - Day - (225)` |
| `CF003 - Active -   Day - (226)` |
| `CF003 - Active - Day - (227)` |
| `CJ001 - Missing   Queen - Day -  (100)` |
| `CJ001 - Missing Queen - Day -  (101)` |
| `CJ001 - Missing   Queen - Day -  (102)` |
| `CJ001 - Missing Queen - Day -  (103)` |
| `CJ001 - Missing   Queen - Day -  (104)` |
| `GH001 - Active - Day - 141022_0659_0751` |
| `Hive1_12_06_2018_QueenBee_H1_audio___15_00_00` |
| `Hive1_12_06_2018_QueenBee_H1_audio___15_10_00` |
| `Hive1_12_06_2018_QueenBee_H1_audio___15_30_00` |
| `Hive1_12_06_2018_QueenBee_H1_audio___15_40_00` |
| `Hive1_12_06_2018_QueenBee_H1_audio___16_50_00` |
| `Hive1_12_06_2018_QueenBee_H1_audio___17_00_00` |
| `Hive1_31_05_2018_NO_QueenBee_H1_audio___15_00_00` |
| `Hive1_31_05_2018_NO_QueenBee_H1_audio___15_20_00` |
| `Hive1_31_05_2018_NO_QueenBee_H1_audio___15_30_00` |
| `Hive1_31_05_2018_NO_QueenBee_H1_audio___15_40_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___15_00_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___15_10_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___15_20_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___15_30_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___15_40_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___16_20_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___16_30_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___16_40_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___16_50_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___17_00_00` |
| `Hive3_15_07_2017_NO_QueenBee_H3_audio___06_10_00` |
| `Hive3_15_07_2017_NO_QueenBee_H3_audio___06_20_00` |
| `Hive3_15_07_2017_NO_QueenBee_H3_audio___06_30_00` |
| `Hive3_15_07_2017_NO_QueenBee_H3_audio___06_40_00` |
| `Hive3_15_07_2017_NO_QueenBee_H3_audio___07_00_00` |
| `Hive3_20_07_2017_QueenBee_H3_audio___06_10_00` |
| `Hive3_20_07_2017_QueenBee_H3_audio___06_20_00` |


The repo experiment with this dataset uses **DUR = 1s**, **SR = 8000Hz** and **THR = 0.1s**.

Using these parameters, the segmentation process produces 24816 one second chunks, of which only 17295 satisfy the threshold and as such will be used for the Queen/NoQueen classification.


## MINI data sets

The **MINI** dataset is a further reduced subset of the reference dataset, using only 2 hives, both at different days for queen and no queen, but all at the same time of the day.

It consists in the following 4 wav audio files

| File Name  (without extension)|
|-|
| `Hive1_12_06_2018_QueenBee_H1_audio___15_00_00` |
| `Hive1_31_05_2018_NO_QueenBee_H1_audio___15_00_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___15_00_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___15_10_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___15_20_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___15_30_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___15_40_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___16_20_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___16_30_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___16_40_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___16_50_00` |
| `Hive3_12_07_2017_NO_QueenBee_H3_audio___17_00_00` |
| `Hive3_15_07_2017_NO_QueenBee_H3_audio___06_10_00` |
| `Hive3_15_07_2017_NO_QueenBee_H3_audio___06_20_00` |
| `Hive3_15_07_2017_NO_QueenBee_H3_audio___06_30_00` |
| `Hive3_15_07_2017_NO_QueenBee_H3_audio___06_40_00` |
| `Hive3_15_07_2017_NO_QueenBee_H3_audio___07_00_00` |
| `Hive3_20_07_2017_QueenBee_H3_audio___06_10_00` |
| `Hive3_20_07_2017_QueenBee_H3_audio___06_20_00` |


The repo experiment with this dataset uses **DUR = 1s**, **SR = 22 050Hz** and **THR = 0.5s**.

Using these parameters, the segmentation process produces 24816 one second chunks, of which only 17295 satisfy the threshold and as such will be used for the Queen/NoQueen classification.
