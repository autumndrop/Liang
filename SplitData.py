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


android_data_name = 'App_data/Android_data.csv'
# android_data_name = 'F:/Liang/App_data/sample.csv'
out_folder_path = 'App_data/individuals'


# # Step 1: Seperate the raw GPS data into individual GPS data files, named with the mobileID
# DataProcessingFunctions_Android.seperateRawDataIntoIndividualData(android_data_name, out_folder_path)

# Step 2: process raw data, generate trip file, location file, sequence file
## Step 2.1: process the raw data, create trip file
namelist=glob.glob('App_data/Individuals/*.csv')
outputPointsFolder = 'App_data/Processed_Data/GPS_points_processed'
outputTripsFolder = 'App_data/Processed_Data/GPS_trips_processed'
outputLocationsFolder = 'App_data/Processed_Data/GPS_locations_processed'
outputDayActivitiesFolder = 'App_datessed_Data/GPS_dayActivities_processed'
outputDaySequenceFolder = 'App_data/Processed_Data/GPS_daySequence_processed'
outputLocationSequenceFolder = 'App_data/Processed_Data/GPS_locationSequence_processed'
tripList = glob.glob('App_data/Processed_Data/GPS_trips_processed/*.csv')


dailyPatternInterval = 20
startHour = 2
locationTimeInterval = 5

if not os.path.isdir(outputPointsFolder):
   os.makedirs(outputPointsFolder)
   os.makedirs(outputTripsFolder)
   os.makedirs(outputLocationsFolder)
   os.makedirs(outputDayActivitiesFolder)
   os.makedirs(outputDaySequenceFolder)
# #
# num_cores = multiprocessing.cpu_count()-2
# #num_cores = 10
# print 'Number of cores', num_cores
# if __name__ == '__main__':
#   Parallel(n_jobs=num_cores)(delayed(DataProcessingFunctions_Android.main_function)(i, outputPointsFolder,outputTripsFolder) for i in namelist)

##Step 2.2 find the min and max lat and long
# locationList = DataProcessingFunctions_Android.findStudyArea(tripList)
# print locationList

##Step 2.3: process the raw data, find stay region, create location file and sequence file
# num_cores = multiprocessing.cpu_count()-2
# if __name__ == '__main__':
#   Parallel(n_jobs=num_cores)(delayed(DataProcessingFunctions_Android.main_function_createLocationSequenceTable)(i, dailyPatternInterval, startHour, outputPointsFolder, outputTripsFolder,outputLocationsFolder,outputDayActivitiesFolder,outputDaySequenceFolder) for i in tripList)

# for i in tripList:
#    DataProcessingFunctions_Android.main_function_createLocationSequenceTable(i, dailyPatternInterval, startHour, outputPointsFolder, outputTripsFolder,outputLocationsFolder,outputDayActivitiesFolder,outputDaySequenceFolder)
# i = 'App_data/test/Individuals/6e5c6063-4a47-4cc5-903b-71fd4939c92e.csv'
# DataProcessingFunctions_Android.main_function_createLocationSequenceTable(i, dailyPatternInterval, startHour, outputPointsFolder, outputTripsFolder,outputLocationsFolder,outputDayActivitiesFolder,outputDaySequenceFolder)

# ##Step 2.4: generating location sequences based on trips
# DataProcessingFunctions_Android.main_createLocationSequence(tripList,outputLocationSequenceFolder,locationTimeInterval, startHour)


# ## Step 2.5: plot location sequence figures
# locationPlotFolderName = 'App_data/Processed_Data/GPS_locationSequence_plots'
# locationSequencelist=glob.glob('App_data/Processed_Data/GPS_locationSequence_processed/*.csv')
# DataProcessingFunctions_Android.main_plotLocationSequence(locationSequencelist,locationPlotFolderName)