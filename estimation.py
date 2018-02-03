import os
import glob
import csv
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime, timedelta, date
import DataProcessingFunctions_Android
reload(DataProcessingFunctions_Android)

import os
from os.path import dirname, abspath
dir_path = dirname(os.path.realpath(__file__))
parent_path = dirname(dirname(dirname(abspath(dir_path))))
print dir_path
print parent_path
os.chdir(parent_path)



# sequenceAlignmentBio(self.daySequence[k],self.daySequence[i])