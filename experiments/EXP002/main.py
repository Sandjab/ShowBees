"""Perform segmentation and labellisation of input annotated dataset, according to parameters"""

#============================================== IMPORTS =============================================

#===== Standard imports
import os
import warnings                   # This block prevents display of harmless warnings, but should be
warnings.filterwarnings("ignore") # commented out till the final version, to avoid missing "real" warnings 

#===== 3rd party imports
# None

#===== Repository imports
import proxycodelib               # Mandatory. Allow access to shared python code in the upper 'codelib' directory
from decomposition import create_chunks

#============================================ PARAMETERS ============================================
CHUNK_DURATION = 1                      # Chunk duration in seconds
SAMPLE_RATE    = 8000                   # Chunk sample rate in Hz, None if you
THRESHOLDS     = [0, 0.1, 0.5, 0.9, 1]  # "No Bee" thresholds list in seconds (one label file per list element)

# Path where to find initial annotated dataset (audio and lab files)
ANNOTATED_DATASET_PATH ="D:\\datasets\\sounds\\MINI" + os.sep 

# Path where to save audio chunks and labels files.
AUDIO_LABELS_PATH = ANNOTATED_DATASET_PATH + 'chunks_' + str(SR) + "Hz_" + str(DUR)+'sec' + os.sep  

#=============================================== MAIN ===============================================

build_chunks( ANNOTATED_DATASET_PATH, AUDIO_LABELS_PATH, CHUNK_DURATION , SAMPLE_RATE, THRESHOLDS )