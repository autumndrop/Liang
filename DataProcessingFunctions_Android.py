# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from datetime import datetime, timedelta, date
from pytz import timezone
import numpy as np
from math import radians, cos, sin, asin, sqrt, ceil
import pytz
import re
import csv
import ast
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import matplotlib
import matplotlib.pyplot as plt
import glob
import operator
import pytimeparse

# from geopy.geocoders import GoogleV3

'''
DataProcessingFunctions.py includes all the data processing functions for Android data and the main_function is the main function. It does the following processes:
    1.    Filter GPS data, keep only satellites = 'D'
    2.    Create long2 = -long
    3.    Process data and time attribte and create datetime field
    4.    Add time and distance different from previous point
    5.    Add a label called moving based on distance, speed and time difference
    6.    And end point, judge whether it's trip end or not
    7.    Add a label called static which means this point is static for sure, it will be to filter static points for accurate location estimation
    8.    Create trips
    9.    Filter trips based on avg_speed and dist_OD
'''


# MIN_LAT = 28.281417
# MIN_LNG = -91.762388
# GRID_LENGTH_LAT = 0.0009
# # 100m is 0.0009 for latitude
# GRID_LENGTH_LNG = 0.001157
# # 100m is 0.001157 for longitude
# ZONE = 20000



MIN_LAT = -37.007159999999999
MIN_LNG = -124.3716
GRID_LENGTH_LAT = 0.0009
# 100m is 0.0009 for latitude
GRID_LENGTH_LNG = 0.001157
# 100m is 0.001157 for longitude
ZONE = 300000

def try_parsing_date(text):
    '''
    Convert a '%Y-%m-%d' type string to datetime object
    Parameters:
    -------------
    text: date string

    Returns:
    -------------
    datetime: datetime object
    '''
    temp = text.split('-')
    if len(temp) == 3:
        return date(int(temp[0]), int(temp[1]), int(temp[2]))
    else:
        temp = text.split('/')
        return date(int(temp[2]), int(temp[0]), int(temp[1]))


def try_parsing_dateTime(text):
    '''
    Convert a '%Y-%m-%d %H:%M:%S' type string to datetime object
    Parameters:
    -------------
    text: datetime string

    Returns:
    -------------
    datetime: datetime object
    '''
    fmt = '%Y-%m-%d %H:%M:%S'
    fmt1 = '%m/%d/%Y %H:%M'
    if len(text) <= 16:
        return datetime.strptime(text, fmt1)
    else:
        return datetime.strptime(text[0:19], fmt)


def try_parsing_time(text):
    '''
    Convert a '%H:%M:%S' type string to datetime object
    Parameters:
    -------------
    text: time string

    Returns:
    -------------
    datetime: datetime object
    '''
    fmt = '%H:%M:%S'
    return datetime.strptime(text, fmt)


def try_parsing_duration(text):
    p1 = text.split(' ')[0]
    p2 = text.split(' ')[2]
    tStr = p2.split('.')[0]
    t = try_parsing_time(tStr)


def addDateTime(a):
    '''Combine the date and time variable to a datetime.datetime variable
    Parameters:
    -------------
    a: pandas.dataframe object

    Returns:
    -------------
    a: original dataframe with an additional column of created date variable with name dateTime'''
    assert isinstance(a, pd.core.frame.DataFrame), 'Argument of wrong type'
    dateString = map(str, a.date.tolist())
    timeInt = map(int, a.time.tolist())
    dayInt = [int(i[:-4]) for i in dateString]
    monthInt = [int(i[-4:-2]) for i in dateString]
    yearInt = [int(i[-2:]) + 2000 for i in dateString]
    secInt = [int(i % 100) for i in timeInt]
    minInt = [int(i / 100 % 100) for i in timeInt]
    hourInt = [int(i / 10000) for i in timeInt]
    dateTimeInt = [
        datetime(yearInt[i], monthInt[i], dayInt[i], hourInt[i], minInt[i], secInt[i], 0, utc).astimezone(eastern) for i
        in range(0, a.shape[0])]
    a['dateTime'] = pd.Series(dateTimeInt)
    return a


def timeDiff(p, a):
    '''
    Calculate the time difference between dateTime of p and dateTime of p-1
    Parameters:
    -------------
    a: pandas.dataframe object
    p: int

    Returns:
    -------------
    timedelta: a.loc[p,'dateTime']-a.loc[p-1,'dateTime']
    '''
    if p == 0:
        return timedelta(minutes=0)
    else:
        return a.loc[p, 'dateTime'] - a.loc[p - 1, 'dateTime']


def timeDiff_Next(p, a):
    '''
    Calculate the time difference between dateTime of p+1 and dateTime of p
    Parameters:
    -------------
    a: pandas.dataframe object
    p: int

    Returns:
    -------------
    if p = len(a) -1, return np.nan;
    otherwise return timedelta: a.loc[p,'dateTime']-a.loc[p-1,'dateTime']
    '''
    if p + 1 > len(a) - 1:
        return timedelta(minutes=-1)
    else:
        return a.loc[p + 1, 'dateTime'] - a.loc[p, 'dateTime']


def timeDiffTwoPoints(p1, p2, a):
    return a.loc[p2, 'dateTime'] - a.loc[p1, 'dateTime']


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Unit: m
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000


def distanceDiff(p, a):
    '''
    Calculate the distance difference between p and p-1
    Parameters:
    -------------
    a: pandas.dataframe object
    p: int

    Returns:
    -------------
    haversine distance, with unit of meter
    '''
    if p == 0:
        return 0
    else:
        return haversine(a.loc[p, 'long2'], a.loc[p, 'lat'], a.loc[p - 1, 'long2'], a.loc[p - 1, 'lat'])


def distanceDiffTwoPoints(p1, p2, a):
    '''
    Calculate the distance difference between p1 and p2
    Parameters:
    -------------
    a: pandas.dataframe object
    p1,p2: int

    Returns:
    -------------
    haversine distance, with unit of meter
    '''
    return haversine(a.loc[p1, 'long2'], a.loc[p1, 'lat'], a.loc[p2, 'long2'], a.loc[p2, 'lat'])


def addTimeDiff(a):
    # add time difference to previous point, delete records that are not consistent in time
    timeD = [timeDiff(i, a) for i in range(0, a.shape[0])]
    a['dateTimeDifference'] = pd.Series(timeD)
    flag = 0
    for i in a.index.values:
        if a.loc[i, 'dateTimeDifference'] < timedelta(minutes=0):
            print a.loc[i, 'dateTimeDifference']
            a = a.drop(i)
            a = a.reset_index(drop=True)
            flag = 1
            del a['dateTimeDifference']
            print 'drop negative duration'
            return addTimeDiff(a)
    if flag != 1:
        return a


def addDistanceDiff(a):
    distanceD = [distanceDiff(i, a) for i in range(0, a.shape[0])]
    a['distanceDifference'] = pd.Series(distanceD)
    return a


def determinIfPointIsMovingUsingDistance(a, i, k):
    if i >= k:
        dist = haversine(a.loc[i, 'long2'], a.loc[i, 'lat'], a.loc[i - k, 'long2'], a.loc[i - k, 'lat'])
        timeD = a.loc[i, 'dateTime'] - a.loc[i - k, 'dateTime']
        if dist > 40 and timeD < timedelta(minutes=10):
            return True
    return False


def determineIfPointIsMoving(a, i):
    move = a.loc[i, 'speed'] > 3 or determinIfPointIsMovingUsingDistance(a, i,
                                                                         1) or determinIfPointIsMovingUsingDistance(a,
                                                                                                                    i,
                                                                                                                    2) or determinIfPointIsMovingUsingDistance(
        a, i, 3)
    return move


def addMoving(a):
    # Determine if a signle point is moving or not compared to previous point
    # move = [ a.loc[i,'speed'] > 0.3 or (a.loc[i,'distanceDifference'] > 10 and a.loc[i,'dateTimeDifference'] < timedelta(minutes = 5)) for i in range(0,a.shape[0])]
    move = [determineIfPointIsMoving(a, i) for i in range(0, a.shape[0])]
    a['moving'] = pd.Series(move)
    return a


def pointNextMoving(a, p):
    # Determine if point p+1 is moving or not
    if p == len(a) - 1:
        return np.nan
    elif a.loc[p + 1, 'moving'] == True:
        return True
    else:
        return False


def moveEndPointBasedOnThreePoints(a, p):
    # Determine if point is end point for the trip or not
    if timeDiff_Next(p, a) >= timedelta(minutes=60) or timeDiff_Next(p, a) < timedelta(minutes=0):
        return True
    elif p + 2 <= len(a) - 1:
        if a.loc[p, 'moving'] == False and a.loc[p + 1, 'moving'] == False and (
                timeDiff_Next(p + 1, a) >= timedelta(minutes=60) or timeDiff_Next(p + 1, a) < timedelta(minutes=0)):
            return True
        elif a.loc[p, 'moving'] == False and a.loc[p + 1, 'moving'] == False and a.loc[p + 2, 'moving'] == False:
            return True
    return False


def findStayPoint(a, timeThred, distanceThred, speedThred):
    a['stayPoint'] = None
    a['stayLat'] = None
    a['stayLng'] = None
    i = 0
    while (i + 1 < a.shape[0]):
        # Assume i is the first point in the stay point, find when the people leave the stay point
        if a.loc[i, 'speed'] < speedThred:
            p1 = i
            p2 = i + 1
            stopFlag = 1
            while (p2 < a.shape[0] and stopFlag):
                if distanceDiffTwoPoints(p1, p2, a) >= distanceThred or a.loc[p2, 'speed'] >= speedThred:
                    stopFlag = 0
                else:
                    p2 = p2 + 1
            p2 = p2 - 1
            if timeDiffTwoPoints(p1, p2, a) >= timeThred:
                a.loc[p1, 'stayPoint'] = 1
                a.loc[p2, 'stayPoint'] = 2
                tempLat = 0
                tempLng = 0
                tempCount = p2 - p1 + 1
                for k in range(tempCount):
                    tempLat = tempLat + a.loc[p1 + k, 'lat']
                    tempLng = tempLng + a.loc[p1 + k, 'long2']
                lat = tempLat / tempCount
                lng = tempLng / tempCount
                for k in range(tempCount):
                    a.loc[p1 + k, 'stayLat'] = lat
                    a.loc[p1 + k, 'stayLng'] = lng
                i = p2 + 1
            else:
                i = p1 + 1
        else:
            i = i + 1
    return a


def addEndPointLabel_stayPoint(a):
    '''
    Identify the starting and ending point of a trip
    Parameters:
    -------------
    a: pandas.dataframe object

    Returns:
    -------------
    a: original dataframe with an additional column of created label with name endPoint
    endPoint = 1 represents trip starting point
    endPoint = 2 represents trip ending point
    endPoint  = 3 represent time gap, before that is stationary
    endPoint = 4 represent end of trip when there is a big gap afterwards or it's end of the record
    C:\Users\LIANG\OneDrive\Travel Pattern Prediction\Code
    '''
    assert isinstance(a, pd.core.frame.DataFrame), 'Argument of wrong type'
    flag = 1
    a['endPoint_lat'] = None
    a['endPoint_lng'] = None
    # flag = 1 reprent strip starting point, 2 represent trip ending point
    i = 0
    while (i + 3 < a.shape[0]):
        if flag == 1:
            # looking for starting point

            p = i
            if i == 0 and pd.isnull(a.loc[p,'stayLat']):
                a.loc[p, 'endPoint'] = flag
                a.loc[p, 'endPoint_lat'] = a.loc[p, 'stayLat']
                a.loc[p, 'endPoint_lng'] = a.loc[p, 'stayLng']
                flag = 2
            else:
                p = p+1
                stopFlag = 0
                while (stopFlag == 0):
                    # Find next point that is moving or jump more than 60 min or to the end of the file
                    if p + 1 >= a.shape[0]:
                        stopFlag = 1
                    elif not pd.isnull(a.loc[p, 'stayPoint']):
                        stopFlag = 1
                    else:
                        p = p + 1
                if p + 1 < a.shape[0]:
                    if not pd.isnull(a.loc[p, 'stayPoint']):
                        a.loc[p, 'endPoint'] = flag
                        a.loc[p, 'endPoint_lat'] = a.loc[p, 'stayLat']
                        a.loc[p, 'endPoint_lng'] = a.loc[p, 'stayLng']
                        flag = 2
            i = p + 1
        else:
            # looking for ending point
            p = i
            stopFlag = 0
            while stopFlag == 0:
                if not pd.isnull(a.loc[p, 'stayPoint']) or p + 1 >= \
                        a.shape[0]:
                    stopFlag = 1
                else:
                    p = p + 1

            a.loc[p, 'endPoint'] = 2
            a.loc[p, 'endPoint_lat'] = a.loc[p, 'stayLat']
            a.loc[p, 'endPoint_lng'] = a.loc[p, 'stayLng']
            flag = 1
            i = p + 1
    return a


def addEndPointLabel(a):
    '''
    Identify the starting and ending point of a trip
    Parameters:
    -------------
    a: pandas.dataframe object

    Returns:
    -------------
    a: original dataframe with an additional column of created label with name endPoint
    endPoint = 1 represents trip starting point
    endPoint = 2 represents trip ending point
    endPoint  = 3 represent time gap, before that is stationary
    endPoint = 4 represent end of trip when there is a big gap afterwards or it's end of the record
    C:\Users\LIANG\OneDrive\Travel Pattern Prediction\Code
    '''
    assert isinstance(a, pd.core.frame.DataFrame), 'Argument of wrong type'
    flag = 1
    # flag = 1 reprent strip starting point, 2 represent trip ending point
    i = 0
    while (i + 3 < a.shape[0]):
        if flag == 1:
            # looking for starting point
            p = i
            while (pointNextMoving(a, p) == False and timeDiff_Next(p, a) < timedelta(minutes=60) and timeDiff_Next(p,
                                                                                                                    a) >= timedelta(
                    minutes=0)):
                # Find next point that is moving or jump more than 60 min or to the end of the file
                p = p + 1
            if timeDiff_Next(p, a) >= timedelta(minutes=60):
                a.loc[p, 'endPoint'] = 3
            elif pointNextMoving(a, p):
                a.loc[p, 'endPoint'] = flag
                flag = 2
            i = p + 1
        else:
            # looking for ending point
            p = i
            while not moveEndPointBasedOnThreePoints(a, p):
                p = p + 1
            if timeDiff_Next(p, a) >= timedelta(minutes=60):
                a.loc[p, 'endPoint'] = 4
                flag = 1
            else:
                a.loc[p, 'endPoint'] = 2
                flag = 1
            i = p + 1
    return a


def addStaticPointLabel(a):
    '''
    This function finds points dont' move and lable it 1 for attribute static
    '''
    # static = [ a.loc[i,'speed'] <= 0.3 and (a.loc[i,'distanceDifference'] <1) for i in range(0,a.shape[0])]
    static = [a.loc[i, 'speed'] <= 0.5 and (a.loc[i, 'distanceDifference'] < 5) for i in range(0, a.shape[0])]
    a['static'] = pd.Series(static)
    return a


def calculateAccurateLatLng(tripFlag, a, n1, n2):
    lat = np.nan
    lng = np.nan
    if tripFlag == 3:
        return [np.nan, np.nan]
    else:
        temp_static_flag = 0
        temp_accurate_lat = 0
        temp_accurate_long = 0
        for i in range(n1, n2 + 1):
            if a.loc[i, 'static']:
                temp_static_flag = temp_static_flag + 1
                temp_accurate_lat = temp_accurate_lat + a.loc[i, 'lat']
                temp_accurate_long = temp_accurate_long + a.loc[i, 'long2']
        if temp_static_flag > 0:
            lat = temp_accurate_lat / temp_static_flag
            lng = temp_accurate_long / temp_static_flag
        elif n1 != n2 and haversine(a.loc[n1, 'long2'], a.loc[n1, 'lat'], a.loc[n2, 'long2'], a.loc[n2, 'lat']) < 100:
            temp_static_flag = 0
            temp_accurate_lat = 0
            temp_accurate_long = 0
            for k in range(n1, n2 + 1):
                temp_static_flag = temp_static_flag + 1
                temp_accurate_lat = temp_accurate_lat + a.loc[k, 'lat']
                temp_accurate_long = temp_accurate_long + a.loc[k, 'long2']
            lat = temp_accurate_lat / temp_static_flag
            lng = temp_accurate_long / temp_static_flag
        return [lat, lng]


def updateDurationUpperBound(df):
    for n in df.index.values:
        if n == len(df) - 1 and df.loc[n, 'missing_Data'] == 0:
            df.loc[n, 'is_duration_D_accurate'] = 1
            df.loc[n, 'is_trip_time_accurate'] = 1
            df.loc[n - 1, 'is_duration_D_accurate'] = 1
            df.loc[n - 1, 'is_trip_time_accurate'] = 1
        elif n > 0 and df.loc[n - 1, 'missing_Data'] == 0:
            tripFlagDetail = df.loc[n, 'tripFlagDetail']
            if tripFlagDetail == 1 or tripFlagDetail == 13:
                df.loc[n - 1, 'is_duration_D_accurate'] = 1
                df.loc[n - 1, 'is_trip_time_accurate'] = 1
            elif tripFlagDetail == 2 and df.loc[n - 1, 'missing_Data'] == 0:
                df.loc[n - 1, 'is_duration_D_accurate'] = 0
                df.loc[n - 1, 'is_trip_time_accurate'] = 1
                df.loc[n - 1, 'duration_D_upperbound'] = df.loc[n, 'end_time'] - df.loc[n - 1, 'end_time']
            elif df.loc[n - 1, 'missing_Data'] == 0:
                df.loc[n - 1, 'is_duration_D_accurate'] = 0
                df.loc[n - 1, 'is_trip_time_accurate'] = 0
                df.loc[n - 1, 'duration_D_upperbound'] = df.loc[n, 'end_time'] - df.loc[n - 1, 'end_time']
                df.loc[n - 1, 'trip_time_upperbound'] = df.loc[n, 'end_time'] - df.loc[n - 1, 'start_time']
    for n in df.index.values:
        if df.loc[n, 'trip_time_upperbound'] == timedelta(minutes=0):
            df.loc[n, 'trip_time_upperbound'] = df.loc[n, 'trip_time']
        if df.loc[n, 'duration_D_upperbound'] == timedelta(minutes=0):
            df.loc[n, 'duration_D_upperbound'] = df.loc[n, 'duration_D']
    return df


def addEachRecord(tripFlagDetail, a, df, n, n1, n2, n2Previous):
    df.loc[n, 'index_O'] = n1
    df.loc[n, 'lat_O'] = a.loc[n1, 'lat']
    df.loc[n, 'long_O'] = a.loc[n1, 'long2']
    df.loc[n, 'index_D'] = n2
    df.loc[n, 'lat_D'] = a.loc[n2, 'lat']
    df.loc[n, 'long_D'] = a.loc[n2, 'long2']
    df.loc[n, 'start_time'] = a.loc[n1, 'dateTime']
    df.loc[n, 'end_time'] = a.loc[n2, 'dateTime']
    df.loc[n, 'weekday'] = a.loc[n1, 'dateTime'].weekday()
    df.loc[n, 'OD_dist'] = haversine(a.loc[n1, 'long2'], a.loc[n1, 'lat'], a.loc[n2, 'long2'], a.loc[n2, 'lat'])
    df.loc[n + 1, 'index_PreviousD'] = n2
    df.loc[n + 1, 'lat_PreviousD'] = a.loc[n2, 'lat']
    df.loc[n + 1, 'long_PreviousD'] = a.loc[n2, 'long2']
    if n == 0:
        df.loc[n, 'duration_O'] = np.nan
        df.loc[n, 'dist_O_PreviousD'] = np.nan
    else:
        temp_time_PreviousD = a.loc[n2Previous, 'dateTime']
        temp_lat_PreviousD = a.loc[n2Previous, 'lat']
        temp_long_PreviousD = a.loc[n2Previous, 'long2']
        df.loc[n, 'duration_O'] = a.loc[n1, 'dateTime'] - temp_time_PreviousD
        df.loc[n - 1, 'duration_D'] = a.loc[n1, 'dateTime'] - temp_time_PreviousD
        df.loc[n, 'dist_O_PreviousD'] = haversine(temp_long_PreviousD, temp_lat_PreviousD, a.loc[n1, 'long2'],
                                                  a.loc[n1, 'lat'])
    df.loc[n, 'trip_time'] = a.loc[n2, 'dateTime'] - a.loc[n1, 'dateTime']
    if tripFlagDetail == 1:
        temp_trip_dist = 0
        temp_max_speed = 0
        temp_avg_speed = 0
        for k in range(n1, n2 + 1):
            temp_trip_dist = temp_trip_dist + a.loc[k, 'distanceDifference']
            temp_max_speed = max(temp_max_speed, a.loc[k, 'speed'])
        temp_avg_speed = temp_trip_dist * 1.0 / df.loc[n, 'trip_time'].total_seconds() * 2.23694
        df.loc[n, 'trip_dist'] = temp_trip_dist
        df.loc[n, 'max_speed'] = temp_max_speed
        df.loc[n, 'avg_speed'] = temp_avg_speed
    # [lat,lng] = calculateAccurateLatLng(tripFlagDetail, a, n2Previous, n1)
    accurateLat = None
    accurateLng = None
    if not pd.isnull(a.loc[n1, 'endPoint_lat']):
        accurateLat = a.loc[n1, 'endPoint_lat']
        accurateLng = a.loc[n1, 'endPoint_lng']
    elif n1 > 0:
        if not pd.isnull(a.loc[n1 - 1, 'endPoint_lat']) and a.loc[n1, 'distanceDifference'] < 300:
            accurateLat = a.loc[n1 - 1, 'endPoint_lat']
            accurateLng = a.loc[n1 - 1, 'endPoint_lng']
    df.loc[n, 'lat_O_accurate'] = accurateLat
    df.loc[n, 'long_O_accurate'] = accurateLng
    df.loc[n, 'lat_PreviousD_accurate'] = accurateLat
    df.loc[n, 'long_PreviousD_accurate'] = accurateLng
    if n - 1 >= 0:
        df.loc[n - 1, 'lat_D_accurate'] = accurateLat
        df.loc[n - 1, 'long_D_accurate'] = accurateLng
    df.loc[n, 'tripFlagDetail'] = tripFlagDetail
    return df


def addRecord(tripFlag, a, df, n, n1, n2, n2Previous):
    # print 'n=',n,'n1=',n1,'n2=',n2
    if tripFlag == 1:
        df = addEachRecord(1, a, df, n, n1, n2, n2Previous)
        n = n + 1
    elif tripFlag == 2:
        df = addEachRecord(2, a, df, n, n1, n2, n2Previous)
        n = n + 1
    elif tripFlag == 3:
        df = addEachRecord(13, a, df, n, n1, n2, n2Previous)
        n1 = n2
        n2Previous = n2
        n2 = n2 + 1
        n = n + 1
        df = addEachRecord(3, a, df, n, n1, n2, n2Previous)
        n = n + 1
    return [n, df]


def deleteLastNullRecord(df):
    n = df.shape[0] - 1
    print 'The type of row ', n, type(df.loc[n, 'duration_D'])
    print 'The value of row ', n, df.loc[n, 'duration_D']
    if df.loc[n, 'duration_D'] == np.nan or df.loc[n, 'duration_D'] == pd.NaT:
        df = df.drop(df.shape[0] - 1)
        return deleteLastNullRecord(df)
    else:
        return df


def deleteTripNoDestinationDuration(df):
    df = df[pd.notnull(df['duration_D'])]
    # df = df.drop(df.shape[0]-1)
    df = deleteLastNullRecord(df)
    return df


def createTripDF(a):
    print 'creatTripDF inside'
    df = pd.DataFrame(np.nan, index=[0],
                      columns=['index_PreviousD', 'lat_PreviousD', 'long_PreviousD', 'index_O', 'lat_O', 'long_O',
                               'index_D', 'lat_D', 'long_D', 'lat_PreviousD_accurate', 'long_PreviousD_accurate',
                               'lat_O_accurate', 'long_O_accurate', 'lat_D_accurate', 'long_D_accurate',
                               'dist_O_PreviousD', 'duration_O', 'duration_D'])
    df['start_time'] = None
    df['end_time'] = None
    df['trip_time'] = None
    df['trip_dist'] = None
    df['max_speed'] = None
    df['OD_dist'] = None
    df['avg_speed'] = None
    df['weekday'] = None
    df['tripFlagDetail'] = None
    i = 0
    n = 0
    n2Previous = 0
    n1 = 0
    n2 = 0
    tripFlag = 0
    while (i < a.shape[0]):
        # print 'i=',i
        n2Previous = n2
        p = i
        findTripFlag = 0
        while p < a.shape[0] and a.loc[p, 'endPoint'] != 1 and a.loc[p, 'endPoint'] != 3:
            p = p + 1
        n1 = p
        # print 'n1=',n1
        if p >= a.shape[0]:
            df = deleteTripNoDestinationDuration(df)
            return df
        if a.loc[p, 'endPoint'] == 3:
            n2 = p + 1
            tripFlag = 2
            findTripFlag = 1
        elif a.loc[p, 'endPoint'] == 1:
            while p < a.shape[0] and a.loc[p, 'endPoint'] != 2 and a.loc[p, 'endPoint'] != 4:
                p = p + 1
            n2 = p
            # print 'n2=',n2
            if p >= a.shape[0]:
                df = deleteTripNoDestinationDuration(df)
                return df
            if a.loc[p, 'endPoint'] == 2:
                tripFlag = 1
                findTripFlag = 1
            elif a.loc[p, 'endPoint'] == 4:
                tripFlag = 3
                findTripFlag = 1
        if findTripFlag == 1:
            [n, df] = addRecord(tripFlag, a, df, n, n1, n2, n2Previous)
            if tripFlag == 1:
                n2Previous = n2
            else:
                n2Previous = n2 + 1
        i = p + 1
    df = deleteTripNoDestinationDuration(df)
    return df


def filterTrip(df):
    '''
    Find trips that caused by error of GPS which have very low average speed and low O to D distance
    Filter trip that travel back
    '''
    n = 0
    while (n < df.shape[0]):
        # if df.loc[n,'avg_speed'] < 2 and (df.loc[n,'OD_dist'] < 60 or df.loc[n,'max_speed']<5) or df.loc[n,'OD_dist'] < 300:
        if df.loc[n, 'missing_Data'] == 0:
            if df.loc[n, 'avg_speed'] < 2 and (df.loc[n, 'OD_dist'] < 60 or df.loc[n, 'max_speed'] < 5):
                if n == 0:
                    df.loc[n + 1, 'duration_O'] = np.nan
                    df.loc[n + 1, 'lat_PreviousD'] = np.nan
                    df.loc[n + 1, 'long_PreviousD'] = np.nan
                    df.loc[n + 1, 'index_PreviousD'] = np.nan
                    df.loc[n + 1, 'lat_PreviousD_accurate'] = np.nan
                    df.loc[n + 1, 'long_PreviousD_accurate'] = np.nan
                elif n == df.shape[0] - 1:
                    df.loc[n - 1, 'duration_D'] = np.nan
                else:
                    df.loc[n - 1, 'duration_D'] = df.loc[n + 1, 'start_time'] - df.loc[n - 1, 'end_time']
                    df.loc[n + 1, 'duration_O'] = df.loc[n + 1, 'start_time'] - df.loc[n - 1, 'end_time']
                    df.loc[n + 1, 'lat_PreviousD'] = df.loc[n - 1, 'lat_D']
                    df.loc[n + 1, 'long_PreviousD'] = df.loc[n - 1, 'long_D']
                    df.loc[n + 1, 'index_PreviousD'] = df.loc[n - 1, 'index_D']
                    df.loc[n + 1, 'lat_PreviousD_accurate'] = df.loc[n - 1, 'lat_D_accurate']
                    df.loc[n + 1, 'long_PreviousD_accurate'] = df.loc[n - 1, 'long_D_accurate']
                df = df.drop(n)
                df = df.reset_index(drop=True)
                n = n - 1
        n = n + 1
    df = deleteTripNoDestinationDuration(df)
    return df


def filterTrip_deleteCircleTrip(df):
    '''
    Delete circle trip
    '''
    n = 0
    while (n < df.shape[0]):
        if df.loc[n, 'locationID_O'] == df.loc[n, 'locationID_D'] and df.loc[n, 'missing_Data'] == 0:
            if n == 0:
                df.loc[n + 1, 'duration_O'] = np.nan
                df.loc[n + 1, 'lat_PreviousD'] = np.nan
                df.loc[n + 1, 'long_PreviousD'] = np.nan
                df.loc[n + 1, 'index_PreviousD'] = np.nan
                df.loc[n + 1, 'lat_PreviousD_accurate'] = np.nan
                df.loc[n + 1, 'long_PreviousD_accurate'] = np.nan
            elif n == df.shape[0] - 1:
                df.loc[n - 1, 'duration_D'] = df.loc[n, 'end_time'] + df.loc[n, 'duration_D'] - df.loc[
                    n - 1, 'end_time']
            else:
                df.loc[n - 1, 'duration_D'] = df.loc[n + 1, 'start_time'] - df.loc[n - 1, 'end_time']
                df.loc[n + 1, 'duration_O'] = df.loc[n + 1, 'start_time'] - df.loc[n - 1, 'end_time']
                df.loc[n + 1, 'lat_PreviousD'] = df.loc[n - 1, 'lat_D']
                df.loc[n + 1, 'long_PreviousD'] = df.loc[n - 1, 'long_D']
                df.loc[n + 1, 'index_PreviousD'] = df.loc[n - 1, 'index_D']
                df.loc[n + 1, 'lat_PreviousD_accurate'] = df.loc[n - 1, 'lat_D_accurate']
                df.loc[n + 1, 'long_PreviousD_accurate'] = df.loc[n - 1, 'long_D_accurate']
            df = df.drop(n)
            df = df.reset_index(drop=True)
            n = n - 1
        n = n + 1
    return df


def filterTrip_deleteCircleTrip_stayRegion(df):
    '''
    Delete circle trip
    '''
    n = 0
    while (n < df.shape[0]):
        if df.loc[n, 'regionID_O'] == df.loc[n, 'regionID']:
            print 'delete ', n
            if n == 0:
                df.loc[n + 1, 'duration_O'] = np.nan
                df.loc[n + 1, 'lat_PreviousD'] = np.nan
                df.loc[n + 1, 'long_PreviousD'] = np.nan
                df.loc[n + 1, 'index_PreviousD'] = np.nan
                df.loc[n + 1, 'lat_PreviousD_accurate'] = np.nan
                df.loc[n + 1, 'long_PreviousD_accurate'] = np.nan
            elif n == df.shape[0] - 1:
                df.loc[n - 1, 'duration_D'] = df.loc[n, 'end_time'] + df.loc[n, 'duration_D'] - df.loc[
                    n - 1, 'end_time']
            else:
                df.loc[n - 1, 'duration_D'] = df.loc[n + 1, 'start_time'] - df.loc[n - 1, 'end_time']
                df.loc[n + 1, 'duration_O'] = df.loc[n + 1, 'start_time'] - df.loc[n - 1, 'end_time']
                df.loc[n + 1, 'lat_PreviousD'] = df.loc[n - 1, 'lat_D']
                df.loc[n + 1, 'long_PreviousD'] = df.loc[n - 1, 'long_D']
                df.loc[n + 1, 'index_PreviousD'] = df.loc[n - 1, 'index_D']
                df.loc[n + 1, 'lat_PreviousD_accurate'] = df.loc[n - 1, 'lat_D_accurate']
                df.loc[n + 1, 'long_PreviousD_accurate'] = df.loc[n - 1, 'long_D_accurate']
            df = df.drop(n)
            df = df.reset_index(drop=True)
            n = n - 1
        n = n + 1
    return df


def parse(s):
    '''
    transfer timedelta string to dictionary of days, hours,minutes,seconds
    '''
    if 'day' in s:
        m = re.match(r'(?P<days>[-\d]+) day[s]* (?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d[\.\d+]*)', s)
    else:
        m = re.match(r'(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d[\.\d+]*)', s)
    return {key: float(val) for key, val in m.groupdict().iteritems()}


def strToTimeDelta(s):
    '''
    construct timedelta object using dictionary of days, hours, minutes and seconds
    '''
    # a = parse(s)
    # a = timedelta(**a)
    seconds = pytimeparse.parse(s)

    return timedelta(seconds=seconds)


def createLocationDF(trip):
    trip1 = trip[np.isfinite(trip['lat_D_accurate'])]
    if trip1.shape[0] == 0:
        locationDF = pd.DataFrame(np.nan, index=[0],
                                  columns=['lat', 'long', 'num', 'total_time', 'avg_time', 'trip_index', 'num_weekday',
                                           'total_time_weekday', 'avg_time_weekday', 'trip_index_weekday',
                                           'num_weekend', 'total_time_weekend', 'avg_time_weekend',
                                           'trip_index_weekend'])
        return locationDF
    if pd.isnull(trip1.iloc[trip1.shape[0] - 1, 17]):
        a = trip1.index.values
        trip1 = trip1.drop(a[trip1.shape[0] - 1])
    if trip1.shape[0] == 0:
        locationDF = pd.DataFrame(np.nan, index=[0],
                                  columns=['lat', 'long', 'num', 'total_time', 'avg_time', 'trip_index', 'num_weekday',
                                           'total_time_weekday', 'avg_time_weekday', 'trip_index_weekday',
                                           'num_weekend', 'total_time_weekend', 'avg_time_weekend',
                                           'trip_index_weekend'])
        return locationDF
    locationDF = pd.DataFrame(np.nan, index=[0],
                              columns=['lat', 'long', 'num', 'total_time', 'avg_time', 'trip_index', 'num_weekday',
                                       'total_time_weekday', 'avg_time_weekday', 'trip_index_weekday', 'num_weekend',
                                       'total_time_weekend', 'avg_time_weekend', 'trip_index_weekend'])
    i = 0
    for k in trip1.index.values:
        if i == 0:
            locationDF.loc[i, 'lat'] = trip1.loc[k, 'lat_D_accurate']
            locationDF.loc[i, 'long'] = trip1.loc[k, 'long_D_accurate']
            if trip1.loc[k, 'weekday'] < 5:
                locationDF.loc[i, 'num_weekday'] = 1
                locationDF.loc[i, 'total_time_weekday'] = trip1.loc[k, 'duration_D']
                locationDF.loc[i, 'avg_time_weekday'] = locationDF.loc[i, 'total_time_weekday'] / locationDF.loc[
                    i, 'num_weekday']
                locationDF.loc[i, 'trip_index_weekday'] = str(int(k))

                locationDF.loc[i, 'num_weekend'] = 0
                locationDF.loc[i, 'total_time_weekend'] = timedelta(minutes=0)
                locationDF.loc[i, 'avg_time_weekend'] = timedelta(minutes=0)
                locationDF.loc[i, 'trip_index_weekend'] = str()
            else:
                locationDF.loc[i, 'num_weekday'] = 0
                locationDF.loc[i, 'total_time_weekday'] = timedelta(minutes=0)
                locationDF.loc[i, 'avg_time_weekday'] = timedelta(minutes=0)
                locationDF.loc[i, 'trip_index_weekday'] = str()

                locationDF.loc[i, 'num_weekend'] = 1
                locationDF.loc[i, 'total_time_weekend'] = trip1.loc[k, 'duration_D']
                locationDF.loc[i, 'avg_time_weekend'] = locationDF.loc[i, 'total_time_weekend'] / locationDF.loc[
                    i, 'num_weekend']
                locationDF.loc[i, 'trip_index_weekend'] = str(int(k))
            locationDF.loc[i, 'num'] = 1
            locationDF.loc[i, 'total_time'] = trip1.loc[k, 'duration_D']
            locationDF.loc[i, 'avg_time'] = locationDF.loc[i, 'total_time'] / locationDF.loc[i, 'num']
            locationDF.loc[i, 'trip_index'] = str(int(k))
            i += 1
        else:
            flag = 0
            for n in range(0, i):
                if haversine(locationDF.loc[n, 'long'], locationDF.loc[n, 'lat'], trip1.loc[k, 'long_D_accurate'],
                             trip1.loc[k, 'lat_D_accurate']) < 300:
                    flag = 1
                    if trip1.loc[k, 'weekday'] < 5:
                        locationDF.loc[n, 'num_weekday'] += 1
                        locationDF.loc[n, 'total_time_weekday'] += trip1.loc[k, 'duration_D']
                        # print type(locationDF.loc[n,'total_time_weekday']),type(trip1.loc[k,'duration_D'])
                        locationDF.loc[n, 'avg_time_weekday'] = locationDF.loc[n, 'total_time_weekday'] / \
                                                                locationDF.loc[n, 'num_weekday']
                        if len(locationDF.loc[n, 'trip_index_weekday']) == 0:
                            locationDF.loc[n, 'trip_index_weekday'] = str(int(k))
                        else:
                            locationDF.loc[n, 'trip_index_weekday'] += '*'
                            locationDF.loc[n, 'trip_index_weekday'] += str(int(k))
                    else:
                        locationDF.loc[n, 'num_weekend'] += 1
                        locationDF.loc[n, 'total_time_weekend'] += trip1.loc[k, 'duration_D']
                        locationDF.loc[n, 'avg_time_weekend'] = locationDF.loc[n, 'total_time_weekend'] / \
                                                                locationDF.loc[n, 'num_weekend']
                        if len(locationDF.loc[n, 'trip_index_weekend']) == 0:
                            locationDF.loc[n, 'trip_index_weekend'] = str(int(k))
                        else:
                            locationDF.loc[n, 'trip_index_weekend'] += '*'
                            locationDF.loc[n, 'trip_index_weekend'] += str(int(k))
                    locationDF.loc[n, 'num'] += 1
                    locationDF.loc[n, 'total_time'] += trip1.loc[k, 'duration_D']
                    locationDF.loc[n, 'avg_time'] = locationDF.loc[n, 'total_time'] / locationDF.loc[n, 'num']
                    if len(locationDF.loc[n, 'trip_index']) == 0:
                        locationDF.loc[n, 'trip_index'] = str(int(k))
                    else:
                        locationDF.loc[n, 'trip_index'] += '*'
                        locationDF.loc[n, 'trip_index'] += str(int(k))
            if flag == 0:
                locationDF.loc[i, 'lat'] = trip1.loc[k, 'lat_D_accurate']
                locationDF.loc[i, 'long'] = trip1.loc[k, 'long_D_accurate']
                if trip1.loc[k, 'weekday'] < 5:
                    locationDF.loc[i, 'num_weekday'] = 1
                    locationDF.loc[i, 'total_time_weekday'] = trip1.loc[k, 'duration_D']
                    locationDF.loc[i, 'avg_time_weekday'] = locationDF.loc[i, 'total_time_weekday'] / locationDF.loc[
                        i, 'num_weekday']
                    locationDF.loc[i, 'trip_index_weekday'] = str(int(k))

                    locationDF.loc[i, 'num_weekend'] = 0
                    locationDF.loc[i, 'total_time_weekend'] = timedelta(minutes=0)
                    locationDF.loc[i, 'avg_time_weekend'] = timedelta(minutes=0)
                    locationDF.loc[i, 'trip_index_weekend'] = str()
                else:
                    locationDF.loc[i, 'num_weekday'] = 0
                    locationDF.loc[i, 'total_time_weekday'] = timedelta(minutes=0)
                    locationDF.loc[i, 'avg_time_weekday'] = timedelta(minutes=0)
                    locationDF.loc[i, 'trip_index_weekday'] = str()

                    locationDF.loc[i, 'num_weekend'] = 1
                    locationDF.loc[i, 'total_time_weekend'] = trip1.loc[k, 'duration_D']
                    locationDF.loc[i, 'avg_time_weekend'] = locationDF.loc[i, 'total_time_weekend'] / locationDF.loc[
                        i, 'num_weekend']
                    locationDF.loc[i, 'trip_index_weekend'] = str(int(k))
                locationDF.loc[i, 'num'] = 1
                locationDF.loc[i, 'total_time'] = trip1.loc[k, 'duration_D']
                locationDF.loc[i, 'avg_time'] = locationDF.loc[i, 'total_time'] / locationDF.loc[i, 'num']
                locationDF.loc[i, 'trip_index'] = str(int(k))
                i += 1
    locationDF = locationDF.sort_values(['total_time'], ascending=[False])
    locationDF = locationDF.reset_index(drop=True)
    return locationDF


def createLocationDFfromCSV(trip):
    trip1 = trip[np.isfinite(trip['lat_D_accurate'])]
    if pd.isnull(trip1.iloc[trip1.shape[0] - 1, 17]):
        a = trip1.index.values
        trip1 = trip1.drop(a[trip1.shape[0] - 1])
    locationDF = pd.DataFrame(np.nan, index=[0],
                              columns=['lat', 'long', 'num', 'total_time', 'avg_time', 'trip_index', 'num_weekday',
                                       'total_time_weekday', 'avg_time_weekday', 'trip_index_weekday', 'num_weekend',
                                       'total_time_weekend', 'avg_time_weekend', 'trip_index_weekend'])
    i = 0
    for k in trip1.index.values:
        if i == 0:
            locationDF.loc[i, 'lat'] = trip1.loc[k, 'lat_D_accurate']
            locationDF.loc[i, 'long'] = trip1.loc[k, 'long_D_accurate']
            if trip1.loc[k, 'weekday'] < 5:
                locationDF.loc[i, 'num_weekday'] = 1
                locationDF.loc[i, 'total_time_weekday'] = strToTimeDelta(trip1.loc[k, 'duration_D'])
                locationDF.loc[i, 'avg_time_weekday'] = locationDF.loc[i, 'total_time_weekday'] / locationDF.loc[
                    i, 'num_weekday']
                locationDF.loc[i, 'trip_index_weekday'] = str(int(k))

                locationDF.loc[i, 'num_weekend'] = 0
                locationDF.loc[i, 'total_time_weekend'] = timedelta(minutes=0)
                locationDF.loc[i, 'avg_time_weekend'] = timedelta(minutes=0)
                locationDF.loc[i, 'trip_index_weekend'] = str()
            else:
                locationDF.loc[i, 'num_weekday'] = 0
                locationDF.loc[i, 'total_time_weekday'] = timedelta(minutes=0)
                locationDF.loc[i, 'avg_time_weekday'] = timedelta(minutes=0)
                locationDF.loc[i, 'trip_index_weekday'] = str()

                locationDF.loc[i, 'num_weekend'] = 1
                locationDF.loc[i, 'total_time_weekend'] = strToTimeDelta(trip1.loc[k, 'duration_D'])
                locationDF.loc[i, 'avg_time_weekend'] = locationDF.loc[i, 'total_time_weekend'] / locationDF.loc[
                    i, 'num_weekend']
                locationDF.loc[i, 'trip_index_weekend'] = str(int(k))
            locationDF.loc[i, 'num'] = 1
            locationDF.loc[i, 'total_time'] = strToTimeDelta(trip1.loc[k, 'duration_D'])
            locationDF.loc[i, 'avg_time'] = locationDF.loc[i, 'total_time'] / locationDF.loc[i, 'num']
            locationDF.loc[i, 'trip_index'] = str(int(k))
            i += 1
        else:
            flag = 0
            for n in range(0, i):
                if haversine(locationDF.loc[n, 'long'], locationDF.loc[n, 'lat'], trip1.loc[k, 'long_D_accurate'],
                             trip1.loc[k, 'lat_D_accurate']) < 300:
                    # use 300 meters as distance buffer
                    flag = 1
                    if trip1.loc[k, 'weekday'] < 5:
                        locationDF.loc[n, 'num_weekday'] += 1
                        locationDF.loc[n, 'total_time_weekday'] += strToTimeDelta(trip1.loc[k, 'duration_D'])
                        locationDF.loc[n, 'avg_time_weekday'] = locationDF.loc[n, 'total_time_weekday'] / \
                                                                locationDF.loc[n, 'num_weekday']
                        if len(locationDF.loc[n, 'trip_index_weekday']) == 0:
                            locationDF.loc[n, 'trip_index_weekday'] = str(int(k))
                        else:
                            locationDF.loc[n, 'trip_index_weekday'] += '*'
                            locationDF.loc[n, 'trip_index_weekday'] += str(int(k))
                    else:
                        locationDF.loc[n, 'num_weekend'] += 1
                        locationDF.loc[n, 'total_time_weekend'] += strToTimeDelta(trip1.loc[k, 'duration_D'])
                        locationDF.loc[n, 'avg_time_weekend'] = locationDF.loc[n, 'total_time_weekend'] / \
                                                                locationDF.loc[n, 'num_weekend']
                        if len(locationDF.loc[n, 'trip_index_weekend']) == 0:
                            locationDF.loc[n, 'trip_index_weekend'] = str(int(k))
                        else:
                            locationDF.loc[n, 'trip_index_weekend'] += '*'
                            locationDF.loc[n, 'trip_index_weekend'] += str(int(k))
                    locationDF.loc[n, 'num'] += 1
                    locationDF.loc[n, 'total_time'] += strToTimeDelta(trip1.loc[k, 'duration_D'])
                    locationDF.loc[n, 'avg_time'] = locationDF.loc[n, 'total_time'] / locationDF.loc[n, 'num']
                    if len(locationDF.loc[n, 'trip_index']) == 0:
                        locationDF.loc[n, 'trip_index'] = str(int(k))
                    else:
                        locationDF.loc[n, 'trip_index'] += '*'
                        locationDF.loc[n, 'trip_index'] += str(int(k))
            if flag == 0:
                locationDF.loc[i, 'lat'] = trip1.loc[k, 'lat_D_accurate']
                locationDF.loc[i, 'long'] = trip1.loc[k, 'long_D_accurate']
                if trip1.loc[k, 'weekday'] < 5:
                    locationDF.loc[i, 'num_weekday'] = 1
                    locationDF.loc[i, 'total_time_weekday'] = strToTimeDelta(trip1.loc[k, 'duration_D'])
                    locationDF.loc[i, 'avg_time_weekday'] = locationDF.loc[i, 'total_time_weekday'] / locationDF.loc[
                        i, 'num_weekday']
                    locationDF.loc[i, 'trip_index_weekday'] = str(int(k))

                    locationDF.loc[i, 'num_weekend'] = 0
                    locationDF.loc[i, 'total_time_weekend'] = timedelta(minutes=0)
                    locationDF.loc[i, 'avg_time_weekend'] = timedelta(minutes=0)
                    locationDF.loc[i, 'trip_index_weekend'] = str()
                else:
                    locationDF.loc[i, 'num_weekday'] = 0
                    locationDF.loc[i, 'total_time_weekday'] = timedelta(minutes=0)
                    locationDF.loc[i, 'avg_time_weekday'] = timedelta(minutes=0)
                    locationDF.loc[i, 'trip_index_weekday'] = str()

                    locationDF.loc[i, 'num_weekend'] = 1
                    locationDF.loc[i, 'total_time_weekend'] = strToTimeDelta(trip1.loc[k, 'duration_D'])
                    locationDF.loc[i, 'avg_time_weekend'] = locationDF.loc[i, 'total_time_weekend'] / locationDF.loc[
                        i, 'num_weekend']
                    locationDF.loc[i, 'trip_index_weekend'] = str(int(k))
                locationDF.loc[i, 'num'] = 1
                locationDF.loc[i, 'total_time'] = strToTimeDelta(trip1.loc[k, 'duration_D'])
                locationDF.loc[i, 'avg_time'] = locationDF.loc[i, 'total_time'] / locationDF.loc[i, 'num']
                locationDF.loc[i, 'trip_index'] = str(int(k))
                i += 1
    locationDF = locationDF.sort_values(['num', 'avg_time'], ascending=[False, False])
    locationDF = locationDF.reset_index(drop=True)
    return locationDF


def addLocationID(tripDF, locationDF):
    # add locationID_O and locationID_D
    tripDF['locationID_O'] = None
    tripDF['locationID_D'] = None
    if locationDF.shape[0] > 0:
        for i in locationDF.index.values:
            if not pd.isnull(locationDF.loc[i, 'trip_index']):
                tripIndex = locationDF.loc[i, 'trip_index'].split('*')
                for k in tripIndex:
                    if k != '':
                        tripDF.loc[int(k), 'locationID_D'] = i
                        if int(k) + 1 in tripDF.index.values:
                            tripDF.loc[int(k) + 1, 'locationID_O'] = i
    return tripDF


def findNearestLocationID(lat1, long1, lat2, long2, locationDF, distanceBuffer):
    for i in locationDF.index.values:
        if not np.isnan(lat1):
            dist1 = haversine(long1, lat1, locationDF.loc[i, 'long'], locationDF.loc[i, 'lat'])
            if dist1 < distanceBuffer:
                return i
        if not np.isnan(lat2):
            dist2 = haversine(long2, lat2, locationDF.loc[i, 'long'], locationDF.loc[i, 'lat'])
            if dist2 < distanceBuffer:
                return i
    return np.nan


def addTripToLocationDF(i_tripDF, tripDF, locationID, locationDF):
    if not np.isnan(locationID):
        locationDF.loc[locationID, 'num'] += 1
        locationDF.loc[locationID, 'total_time'] += tripDF.loc[i_tripDF, 'duration_D']
        print 'Total_time:', locationDF.loc[locationID, 'total_time']
        print 'num:', locationDF.loc[locationID, 'num']
        locationDF.loc[locationID, 'avg_time'] = locationDF.loc[locationID, 'total_time'] / locationDF.loc[
            locationID, 'num']
        locationDF.loc[locationID, 'trip_index'] += '*'
        locationDF.loc[locationID, 'trip_index'] += str(i_tripDF)
        if tripDF.loc[i_tripDF, 'weekday'] < 5:
            locationDF.loc[locationID, 'num_weekday'] += 1
            locationDF.loc[locationID, 'total_time_weekday'] += tripDF.loc[i_tripDF, 'duration_D']
            locationDF.loc[locationID, 'avg_time_weekday'] = locationDF.loc[locationID, 'total_time_weekday'] / \
                                                             locationDF.loc[locationID, 'num_weekday']
            locationDF.loc[locationID, 'trip_index_weekday'] += '*'
            locationDF.loc[locationID, 'trip_index_weekday'] += str(i_tripDF)
        else:
            locationDF.loc[locationID, 'num_weekend'] += 1
            locationDF.loc[locationID, 'total_time_weekend'] += tripDF.loc[i_tripDF, 'duration_D']
            locationDF.loc[locationID, 'avg_time_weekend'] = locationDF.loc[locationID, 'total_time_weekend'] / \
                                                             locationDF.loc[locationID, 'num_weekend']
            locationDF.loc[locationID, 'trip_index_weekend'] += '*'
            locationDF.loc[locationID, 'trip_index_weekend'] += str(i_tripDF)
    return


def matchPointToLocationID(tripDF, locationDF, distanceBuffer):
    for i in tripDF.index.values:
        # Fill the locationID_O
        if pd.isnull(tripDF.loc[i, 'locationID_O']):
            if not pd.isnull(tripDF.loc[i, 'lat_PreviousD_accurate']):
                lat_PreviousD = tripDF.loc[i, 'lat_PreviousD_accurate']
                long_PreviousD = tripDF.loc[i, 'long_PreviousD_accurate']
            else:
                lat_PreviousD = tripDF.loc[i, 'lat_PreviousD']
                long_PreviousD = tripDF.loc[i, 'long_PreviousD']

            if not pd.isnull(tripDF.loc[i, 'lat_O_accurate']):
                lat_O = tripDF.loc[i, 'lat_O_accurate']
                long_O = tripDF.loc[i, 'long_O_accurate']
            else:
                lat_O = tripDF.loc[i, 'lat_O']
                long_O = tripDF.loc[i, 'long_O']
            templocationID = findNearestLocationID(lat_PreviousD, long_PreviousD, lat_O, long_O, locationDF,
                                                   distanceBuffer)
            tripDF.loc[i, 'locationID_O'] = templocationID
            # Fill the locationID_D
            if i > 0:
                tripDF.loc[i - 1, 'locationID_D'] = templocationID
                addTripToLocationDF(i - 1, tripDF, templocationID, locationDF)
    return tripDF


def daterange(d1, d2):
    return [d1 + timedelta(days=i) for i in range((d2 - d1).days + 1)]


def createDayActivityDF(tripDF):
    time1 = tripDF.loc[0, 'start_time']
    date1 = time1.date()
    time2 = tripDF.loc[tripDF.shape[0] - 1, 'start_time']
    date2 = time2.date()
    date_index = daterange(date1, date2)
    dayActivityDF = pd.DataFrame(0, index=pd.Series(date_index),
                                 columns=['num_trips', 'duration', 'num_trips_home', 'duration_home', 'num_trips_work',
                                          'duration_work', 'num_trips_other', 'duration_other', 'weekday'])
    dayActivityDF.duration = np.nan
    dayActivityDF.duration_home = np.nan
    dayActivityDF.duration_work = np.nan
    dayActivityDF.duration_other = np.nan
    for i in tripDF.index.values:
        tempDate = tripDF.loc[i, 'start_time'].date()
        dateIndex = (dayActivityDF.index.values == tempDate)
        tempDuration = tripDF.loc[i, 'duration_D']
        if pd.isnull(tempDuration):
            tempDuration = timedelta(minutes=0)
        dayActivityDF.loc[dateIndex, 'num_trips'] += 1
        if pd.isnull(dayActivityDF.loc[dateIndex, 'duration'])[0]:
            dayActivityDF.loc[dateIndex, 'duration'] = tempDuration
        else:
            dayActivityDF.loc[dateIndex, 'duration'][0] += tempDuration
        if tripDF.loc[i, 'locationID_D'] == 0:
            dayActivityDF.loc[dateIndex, 'num_trips_home'] += 1
            if pd.isnull(dayActivityDF.loc[dateIndex, 'duration_home'])[0]:
                dayActivityDF.loc[dateIndex, 'duration_home'] = tempDuration
            else:
                dayActivityDF.loc[dateIndex, 'duration_home'][0] += tempDuration
        elif tripDF.loc[i, 'locationID_D'] == 1:
            dayActivityDF.loc[dateIndex, 'num_trips_work'] += 1
            if pd.isnull(dayActivityDF.loc[dateIndex, 'duration_work'])[0]:
                dayActivityDF.loc[dateIndex, 'duration_work'] = tempDuration
            else:
                dayActivityDF.loc[dateIndex, 'duration_work'][0] += tempDuration
        else:
            dayActivityDF.loc[dateIndex, 'num_trips_other'] += 1
            if pd.isnull(dayActivityDF.loc[dateIndex, 'duration_other'])[0]:
                dayActivityDF.loc[dateIndex, 'duration_other'] = tempDuration
            else:
                dayActivityDF.loc[dateIndex, 'duration_other'][0] += tempDuration

    for i in dayActivityDF.index.values:
        dayActivityDF.loc[i, 'weekday'] = i.weekday()
    return dayActivityDF


def sequenceCreate(seq, n, loc):
    n = int(n)
    if pd.isnull(loc):
        loc = -1
    if n == 0:
        n = 1
    for i in range(0, n):
        seq.append(loc)
    return seq


def sequenceCreate_L(seq, n, loc):
    n = int(n)
    if pd.isnull(loc):
        loc = -1
    for i in range(0, n):
        seq.append(loc)
    return seq


def daySequenceCreate(tripDF, outFile, dailyPatternInterval, startHour):
    '''
    Create daySequence based on tripDF
    '''
    # currentTime is used to denote the current time for calculation
    # Initialize some variables
    dayIndexes = []
    daySequences = []
    firstFlag = 0
    unobserved_LocID = -1
    trip_ID = -99
    missing_ID = -100
    # Loop through each record in the tripDF
    for i in tripDF.index.values:
        # print i
        # determine TempTripID
        tempTrip_ID = trip_ID
        # get the location ID
        location_ID = tripDF.loc[i, 'locationID_O']
        if pd.isnull(location_ID):
            location_ID = unobserved_LocID
            unobserved_LocID = unobserved_LocID - 1
        if firstFlag == 0:
            # If it is the first trip of the people
            # 1. Figure out the corresponding start date and time
            daySeq = []
            tripStartTime = tripDF.loc[i, 'start_time']
            dayStartTime = startDateTime_daySequence(startHour, tripStartTime)
            currentTime = dayStartTime
            tripEndTime = tripDF.loc[i, 'end_time']
            # print 'DayStartTime:',dayStartTime,'currentTime',currentTime, 'tripStartTime:',tripStartTime,'tripEndTime',tripEndTime
            [daySeq, dayIndexes, daySequences, dayStartTime,
             currentTime] = createSequencesForEachTripRecord_daySequence(currentTime, dayStartTime, tripStartTime,
                                                                         tripEndTime, missing_ID, tempTrip_ID,
                                                                         dayIndexes, daySequences, daySeq,
                                                                         dailyPatternInterval)
            # print 'daySeq,',len(daySeq),'dayStartTime',dayStartTime,'currentTime',currentTime
            # print 'daySeq,',daySeq,'dayStartTime',dayStartTime,'currentTime',currentTime
            firstFlag = 1
        else:
            tripStartTime = tripDF.loc[i, 'start_time']
            tripEndTime = tripDF.loc[i, 'end_time']
            # if it is not the first record
            # print 'DayStartTime:',dayStartTime,'currentTime',currentTime, 'tripStartTime:',tripStartTime,'tripEndTime',tripEndTime
            [daySeq, dayIndexes, daySequences, dayStartTime,
             currentTime] = createSequencesForEachTripRecord_daySequence(currentTime, dayStartTime, tripStartTime,
                                                                         tripEndTime, location_ID, tempTrip_ID,
                                                                         dayIndexes, daySequences, daySeq,
                                                                         dailyPatternInterval)
            # print 'daySeq,',len(daySeq),'dayStartTime',dayStartTime,'currentTime',currentTime
            # print 'daySeq,',daySeq,'dayStartTime',dayStartTime,'currentTime',currentTime
    daySequence = pd.Series(data=daySequences, index=dayIndexes)
    daySequence.to_csv(outFile)
    # for i in range(len(daySequences)):
    #    print dayIndexes[i],len(daySequences[i])
    # print daySequences[i]
    return daySequence


def startDateTime_daySequence(startHour, tripStartTime):
    tempDate = tripStartTime.date()
    # dayStartTime = eastern.localize( datetime(int(tempDate.year), int(tempDate.month), int(tempDate.day), int(startHour)))
    dayStartTime = datetime(int(tempDate.year), int(tempDate.month), int(tempDate.day), int(startHour))
    if dayStartTime > tripStartTime:
        dayStartTime -= timedelta(days=1)
    return dayStartTime


def zeroDuration(dt):
    # determine if a duration is exactly 0
    # print 'duration',dt
    if dt == timedelta(seconds=0):
        # print '0 duration'
        return True
    else:
        return False


def createSequencesForEachTripRecord_daySequence(currentTime, dayStartTime, tripStartTime, tripEndTime, location_ID,
                                                 trip_ID, dayIndexes, daySequences, daySeq, dailyPatternInterval):
    # if tripStartTime is not in the same day as the dayStartTime
    # Parameters need to be updated: currentTime, dayStartTime, tripStartTime, dayIndexes, daySequences, daySeq
    if tripStartTime > dayStartTime + timedelta(days=1):
        originTime = dayStartTime + timedelta(days=1) - currentTime
        n1 = round(originTime.total_seconds() * 1.0 / (dailyPatternInterval * 60))
        daySeq = sequenceCreate(daySeq, n1, location_ID)
        dayIndexes.append(dayStartTime.date())
        daySequences.append(daySeq)
        daySeq = []
        dayStartTime = dayStartTime + timedelta(days=1);
        currentTime = dayStartTime
        return createSequencesForEachTripRecord_daySequence(currentTime, dayStartTime, tripStartTime, tripEndTime,
                                                            location_ID, trip_ID, dayIndexes, daySequences, daySeq,
                                                            dailyPatternInterval)
    elif tripEndTime < dayStartTime + timedelta(days=1):
        # if tripEndTime is before the dayStartTime+1 day, we can close the loop
        [daySeq, currentTime] = addDaySeqForShortTrip_daySequence(currentTime, dayStartTime, tripStartTime, tripEndTime,
                                                                  location_ID, trip_ID, daySeq, dailyPatternInterval)
        return [daySeq, dayIndexes, daySequences, dayStartTime, currentTime]
    else:
        # tripEndTime is after the dayStartTime+1
        originTime = tripStartTime - currentTime
        if not zeroDuration(originTime):
            n1 = originTime.total_seconds() / (dailyPatternInterval * 60)
            daySeq = sequenceCreate(daySeq, n1, location_ID)
        currentTime = tripStartTime
        travelTime = dayStartTime + timedelta(days=1) - currentTime
        n2 = round(travelTime.total_seconds() * 1.0 / (dailyPatternInterval * 60))
        daySeq = sequenceCreate(daySeq, n2, trip_ID)
        dayIndexes.append(dayStartTime.date())
        daySequences.append(daySeq)
        dayStartTime = dayStartTime + timedelta(days=1);
        currentTime = dayStartTime
        tripStartTime = currentTime
        daySeq = []
        return createSequencesForEachTripRecord_daySequence(currentTime, dayStartTime, tripStartTime, tripEndTime,
                                                            location_ID, trip_ID, dayIndexes, daySequences, daySeq,
                                                            dailyPatternInterval)


def addDaySeqForShortTrip_daySequence(currentTime, dayStartTime, tripStartTime, tripEndTime, location_ID, trip_ID,
                                      daySeq, dailyPatternInterval):
    '''
    For trip that within a day
    '''
    originTime = tripStartTime - currentTime
    if not zeroDuration(originTime):
        n1 = round(originTime.total_seconds() * 1.0 / (dailyPatternInterval * 60))
        daySeq = sequenceCreate(daySeq, n1, location_ID)
    currentTime = tripStartTime
    travelTime = tripEndTime - currentTime
    n2 = round(travelTime.total_seconds() * 1.0 / (dailyPatternInterval * 60))
    daySeq = sequenceCreate(daySeq, n2, trip_ID)
    currentTime = tripEndTime
    return [daySeq, currentTime]


def locationSequenceCreate(tripDFName, outFile, locationTimeInterval, startHour):
    '''
    Record a location every certain time interval (locationTimeInterval)
    '''
    # currentTime is used to denote the current time(go to the previous locationTimeInterval) for calculation
    # Initialize some variables
    # print 'locationSequenceCreate'
    tripDF = readTripsFromCSV(tripDFName)
    locationTimeInterval = int(locationTimeInterval)
    dayIndexes = []
    daySequences = []
    firstFlag = 0
    unobserved_LocID = -1
    trip_ID = -99
    missing_ID = -100
    leftSec = 0
    # Loop through each record in the tripDF
    for i in tripDF.index.values:
        # print i
        tempTrip_ID = trip_ID
        # get the location ID
        location_ID = tripDF.loc[i, 'locationID_O']
        if pd.isnull(location_ID):
            location_ID = unobserved_LocID
            unobserved_LocID = unobserved_LocID - 1
        if firstFlag == 0:
            # If it is the first trip of the people
            # 1. Figure out the corresponding start date and time
            daySeq = []
            tripStartTime = tripDF.loc[i, 'start_time']
            dayStartTime = startDateTime(startHour, tripStartTime)
            currentTime = dayStartTime
            tripEndTime = tripDF.loc[i, 'end_time']
            # print 'DayStartTime:',dayStartTime,'currentTime',currentTime, 'tripStartTime:',tripStartTime,'tripEndTime',tripEndTime
            [daySeq, dayIndexes, daySequences, dayStartTime, currentTime] = createSequencesForEachTripRecord(
                currentTime, dayStartTime, tripStartTime, tripEndTime, missing_ID, tempTrip_ID, dayIndexes,
                daySequences, daySeq, locationTimeInterval)
            # print 'daySeq,',len(daySeq),'dayStartTime',dayStartTime,'currentTime',currentTime
            # print 'daySeq,',daySeq,'dayStartTime',dayStartTime,'currentTime',currentTime
            firstFlag = 1
        else:
            tripStartTime = tripDF.loc[i, 'start_time']
            tripEndTime = tripDF.loc[i, 'end_time']
            # if it is not the first record
            # print 'DayStartTime:',dayStartTime,'currentTime',currentTime, 'tripStartTime:',tripStartTime,'tripEndTime',tripEndTime
            [daySeq, dayIndexes, daySequences, dayStartTime, currentTime] = createSequencesForEachTripRecord(
                currentTime, dayStartTime, tripStartTime, tripEndTime, location_ID, tempTrip_ID, dayIndexes,
                daySequences, daySeq, locationTimeInterval)
            # print 'daySeq,',len(daySeq),'dayStartTime',dayStartTime,'currentTime',currentTime
            # print 'daySeq,',daySeq,'dayStartTime',dayStartTime,'currentTime',currentTime
    locationSequence = pd.Series(data=daySequences, index=dayIndexes)
    locationSequence.to_csv(outFile)
    # for i in range(len(daySequences)):
    #    print dayIndexes[i],len(daySequences[i])
    # print daySequences[i]
    return locationSequence


def startDateTime(startHour, tripStartTime):
    tempDate = tripStartTime.date()
    dayStartTime = datetime(tempDate.year, tempDate.month, tempDate.day, startHour)
    if dayStartTime > tripStartTime:
        dayStartTime -= timedelta(days=1)
    return dayStartTime


def createSequencesForEachTripRecord(currentTime, dayStartTime, tripStartTime, tripEndTime, location_ID, trip_ID,
                                     dayIndexes, daySequences, daySeq, locationTimeInterval):
    # if tripStartTime is not in the same day as the dayStartTime
    # Parameters need to be updated: currentTime, dayStartTime, tripStartTime, dayIndexes, daySequences, daySeq
    if tripStartTime > dayStartTime + timedelta(days=1):
        originTime = dayStartTime + timedelta(days=1) - currentTime
        n1 = originTime.total_seconds() / (locationTimeInterval * 60)
        daySeq = sequenceCreate_L(daySeq, n1, location_ID)
        dayIndexes.append(dayStartTime.date())
        daySequences.append(daySeq)
        daySeq = []
        dayStartTime = dayStartTime + timedelta(days=1);
        currentTime = dayStartTime
        return createSequencesForEachTripRecord(currentTime, dayStartTime, tripStartTime, tripEndTime, location_ID,
                                                trip_ID, dayIndexes, daySequences, daySeq, locationTimeInterval)
    elif tripEndTime < dayStartTime + timedelta(days=1):
        # if tripEndTime is before the dayStartTime+1 day, we can close the loop
        [daySeq, currentTime] = addDaySeqForShortTrip(currentTime, dayStartTime, tripStartTime, tripEndTime,
                                                      location_ID, trip_ID, daySeq, locationTimeInterval)
        return [daySeq, dayIndexes, daySequences, dayStartTime, currentTime]
    else:
        # tripEndTime is after the dayStartTime+1
        originTime = tripStartTime - currentTime
        n1 = originTime.total_seconds() / (locationTimeInterval * 60)
        daySeq = sequenceCreate_L(daySeq, n1, location_ID)
        currentTime = calculateCurrentTime(currentTime, locationTimeInterval, originTime)
        travelTime = dayStartTime + timedelta(days=1) - currentTime
        n2 = travelTime.total_seconds() / (locationTimeInterval * 60)
        daySeq = sequenceCreate_L(daySeq, n2, trip_ID)
        dayIndexes.append(dayStartTime.date())
        daySequences.append(daySeq)
        dayStartTime = dayStartTime + timedelta(days=1);
        currentTime = dayStartTime
        tripStartTime = currentTime
        daySeq = []
        return createSequencesForEachTripRecord(currentTime, dayStartTime, tripStartTime, tripEndTime, location_ID,
                                                trip_ID, dayIndexes, daySequences, daySeq, locationTimeInterval)


def addDaySeqForShortTrip(currentTime, dayStartTime, tripStartTime, tripEndTime, location_ID, trip_ID, daySeq,
                          locationTimeInterval):
    '''
    For trip that within a day
    '''
    originTime = tripStartTime - currentTime
    n1 = originTime.total_seconds() / (locationTimeInterval * 60)
    daySeq = sequenceCreate_L(daySeq, n1, location_ID)
    currentTime = calculateCurrentTime(currentTime, locationTimeInterval, originTime)
    travelTime = tripEndTime - currentTime
    n2 = travelTime.total_seconds() / (locationTimeInterval * 60)
    daySeq = sequenceCreate_L(daySeq, n2, trip_ID)
    currentTime = calculateCurrentTime(currentTime, locationTimeInterval, travelTime)
    return [daySeq, currentTime]


def calculateCurrentTime(previousTime, locationTimeInterval, duration):
    '''
    calculate update the currentTime
    '''
    lefSec = duration.total_seconds() % (locationTimeInterval * 60)
    currentTime = previousTime + timedelta(seconds=(duration.total_seconds() - lefSec));
    return currentTime


def readTripsFromCSV(name):
    tripDF = pd.read_csv(name, index_col=0)
    # Need to convert the following String columns:
    # duration_O, duration_D, start_time, end_time, trip_time
    if 'duration_O' in tripDF.columns.values:
        tripDF['duration_O_str'] = tripDF['duration_O']
    tripDF['duration_D_str'] = tripDF['duration_D']
    tripDF['trip_time_str'] = tripDF['trip_time']
    tripDF['start_time_str'] = tripDF['start_time']
    tripDF['end_time_str'] = tripDF['end_time']
    # tripDF['duration_D_upperbound_str'] = tripDF['duration_D_upperbound']
    # tripDF['trip_time_upperbound_str'] = tripDF['trip_time_upperbound']
    # print tripDF.loc[0,'trip_time_str']
    tripDF.loc[0, 'trip_time'] = strToTimeDelta(tripDF.loc[0, 'trip_time_str'])
    for i in tripDF.index.values:
        # print tripDF.loc[i,'trip_time_upperbound']
        if 'duration_O' in tripDF.columns.values:
            if not pd.isnull(tripDF.loc[i, 'duration_O']):
                tripDF.loc[i, 'duration_O'] = strToTimeDelta(tripDF.loc[i, 'duration_O_str'])
        if not pd.isnull(tripDF.loc[i, 'duration_D']):
            tripDF.loc[i, 'duration_D'] = strToTimeDelta(tripDF.loc[i, 'duration_D_str'])
        if not pd.isnull(tripDF.loc[i, 'trip_time']):
            tripDF.loc[i, 'trip_time'] = strToTimeDelta(tripDF.loc[i, 'trip_time_str'])
        if not pd.isnull(tripDF.loc[i, 'start_time']):
            tripDF.loc[i, 'start_time'] = try_parsing_dateTime(tripDF.loc[i, 'start_time_str'])
        if not pd.isnull(tripDF.loc[i, 'end_time']):
            tripDF.loc[i, 'end_time'] = try_parsing_dateTime(tripDF.loc[i, 'end_time_str'])
        # if not pd.isnull(tripDF.loc[i, 'duration_D_upperbound']) and tripDF.loc[i, 'duration_D_upperbound'] != '':
        #     tripDF.loc[i, 'duration_D_upperbound'] = strToTimeDelta(tripDF.loc[i, 'duration_D_upperbound_str'])
        # if not pd.isnull(tripDF.loc[i, 'trip_time_upperbound']) and tripDF.loc[i, 'trip_time_upperbound'] != '':
        #     tripDF.loc[i, 'trip_time_upperbound'] = strToTimeDelta(tripDF.loc[i, 'trip_time_upperbound_str'])
    if 'duration_O' in tripDF.columns.values:
        del tripDF['duration_O_str']
    del tripDF['duration_D_str']
    del tripDF['trip_time_str']
    del tripDF['start_time_str']
    del tripDF['end_time_str']
    # del tripDF['duration_D_upperbound_str']
    # del tripDF['trip_time_upperbound_str']
    return tripDF


def readDaySequenceFromCSV(name):
    dayIndex = []
    daySequence = []
    f = open(name, 'r')
    reader = csv.reader(f)
    for row in reader:
        dayIndex.append(try_parsing_date(row[0]))
        daySequence.append(ast.literal_eval(row[1]))
    f.close()
    return pd.Series(daySequence, index=dayIndex)


def getNumberPrecipitation(x):
    if x == 'T':
        return 0.005
    else:
        return float(x)


def readWeatherDaySequenceFromCSV(name):
    dayIndex = []
    daySequence = []
    dayPrecipitation = []
    df = pd.read_csv(name, index_col=0)
    for i in df.index.values:
        dayIndex.append(try_parsing_date(i))
        daySequence.append(ast.literal_eval(df.loc[i, 'seq']))
        dayPrecipitation.append(getNumberPrecipitation(df.loc[i, 'PrecipitationIn']))
    return [pd.Series(daySequence, index=dayIndex), pd.Series(dayPrecipitation, index=dayIndex)]


def precipitationComparison(p1, p2):
    # compare between different precipitation values and determine if they belongs to the same category
    # Currently if they are both 0, return 1, if they both >0, return 2, otherwise, return 0
    if p1 == 0 and p2 == 0:
        return 1
    elif p1 > 0 and p2 > 0:
        return 2
    else:
        return 0


def genericSequenceAlignment(seq1, seq2):
    a = Sequence(seq1)
    b = Sequence(seq2)
    print 'Sequence A:', a
    print 'Sequence B:', b
    print

    # Create a vocabulary and encode the sequences.
    v = Vocabulary()
    aEncoded = v.encodeSequence(a)
    bEncoded = v.encodeSequence(b)
    print 'Encoded A:', aEncoded
    print 'Encoded B:', bEncoded
    print

    # Create a scoring and align the sequences using global aligner.
    scoring = SimpleScoring(2, -1)
    aligner = GlobalSequenceAligner(scoring, -2)
    score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)

    # Iterate over optimal alignments and print them.
    for encoded in encodeds:
        alignment = v.decodeSequenceAlignment(encoded)
        print alignment
        print 'Alignment score:', alignment.score
        print 'Percent identity:', alignment.percentIdentity()
        print
    return alignment.score, alignment.percentIdentity()


def sequenceAlignmentBio(seq1, seq2):
    '''
    Find the best global alignment between the two sequences.
    Identical characters are given 2 points,
    1 point is deducted for each non-identical character.
    0.5 points are deducted when opening a gap,
    and 0.1 points are deducted when extending it.
    :param seq1: Day sequence 1
    :param seq2: Day sequence 2
    :return: The similarity score of the two sequence
    '''
    combined = seq1 + seq2
    myset = set(combined)
    myset = list(myset)
    chaList = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '`',
               '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '=', '[', '{', ']', '}', ';', ':', '<',
               '>', ',', '.', '?', '/']
    myDict = {}
    for i in range(len(myset)):
        myDict[myset[i]] = chaList[i]
    seq1Char = [myDict[x] for x in seq1]
    seq2Char = [myDict[x] for x in seq2]
    seq1Str = ''.join(map(str, seq1Char))
    seq2Str = ''.join(map(str, seq2Char))
    aList = pairwise2.align.globalms(seq1Str, seq2Str, 2, -1, -.5, -.1)
    return aList[0][2]


def main_function(inputfilename, writeFolderName1, writeFolderName2):
    '''
    Step1: process the raw data (main_function)
        Filter and pre-process GPS data
        For each GPS point, determine if it’s moving or not based on speed, distance from pre. point and time difference
        Determine if a point is trip end or not, save as point table (*_points.csv, save in Processed_Data/GPS_points_processed)
        Extract trip information and create trip table (*_trips.csv, save in Processed_Data/GPS_trips_processed)
        Filter trips, eliminated GPS random error
        Store accurate destination locations and create location table, locations within 300 meters are considered the same, rank the locations and assign location ID (*_locations.csv, save in Processed_Data/GPS_locations_processed)
        Determine home and work location
        Create daily activity table (*_daySummary.csv, save in Processed_Data/GPS_dayActivities_processed)
        Create dailySequence table (*_daySequence.csv, save in Processed_Data/GPS_daySequence_processed)

    '''

    filename = os.path.basename(inputfilename).split('.')[0]
    print filename
    outputName1 = writeFolderName1 + '/' + filename + '_points.csv'
    if os.path.isfile(outputName1):
        print 'skip'
        return
    outputName2 = writeFolderName2 + '/' + filename + '_trips.csv'
    df = pd.read_csv(inputfilename, sep=";", header=None,
                     names=["mobileID", "lat", "long2", "speed", "accuracy", "dateTime"], parse_dates=['dateTime'])
    if len(df)<10:
        print 'skip'
        return
    df = df.sort_values(['dateTime'], ascending=[True])
    df = df[df['accuracy'] < 200]
    startDate = datetime(2015, 1, 1, 0)
    df = df[df['dateTime'] > startDate]
    df = df.reset_index(drop=True)
    a = df.iloc[0,5]
    b = df.iloc[-1,5]
    timeSpan = b - a
    if(timeSpan.days < 14):
        return
    df = addTimeDiff(df)
    df = addDistanceDiff(df)
    # timeThred = timedelta(minutes=20)
    # distanceThred = 500
    # speedThred = 5
    timeThred = timedelta(minutes=20)
    distanceThred = 200
    speedThred = 5
    print 'findStayPoint'
    df = findStayPoint(df, timeThred, distanceThred, speedThred)
    print 'addEndPointLabel'
    df = addEndPointLabel_stayPoint(df)
    df.to_csv(outputName1)
    print 'createTripDF'
    tripDF = createTripDF(df)
    if len(tripDF) > 10:
        tripDF.to_csv(outputName2)
    return


def updateLocationDF(tripDF, locationDF_complete):
    for i in locationDF_complete.index.values:
        locationDF_complete.loc[i, 'num'] = 0
        locationDF_complete.loc[i, 'total_time'] = timedelta(minutes=0)
        locationDF_complete.loc[i, 'avg_time'] = timedelta(minutes=0)
        locationDF_complete.loc[i, 'trip_index'] = str()
        locationDF_complete.loc[i, 'num_weekend'] = 0
        locationDF_complete.loc[i, 'total_time_weekend'] = timedelta(minutes=0)
        locationDF_complete.loc[i, 'avg_time_weekend'] = timedelta(minutes=0)
        locationDF_complete.loc[i, 'trip_index_weekend'] = str()
        locationDF_complete.loc[i, 'num_weekday'] = 0
        locationDF_complete.loc[i, 'total_time_weekday'] = timedelta(minutes=0)
        locationDF_complete.loc[i, 'avg_time_weekday'] = timedelta(minutes=0)
        locationDF_complete.loc[i, 'trip_index_weekday'] = str()
    for i in tripDF.index.values:
        if not pd.isnull(tripDF.loc[i, 'locationID_D']):
            addTripToLocationDF(i, tripDF, tripDF.loc[i, 'locationID_D'], locationDF_complete)
    return locationDF_complete


def main_function_list(namelist, dailyPatternInterval, startHour, outputPointsFolder, outputTripsFolder,
                       outputLocationsFolder, outputDayActivitiesFolder, outputDaySequenceFolder):
    if not os.path.isdir(outputPointsFolder):
        os.makedirs(outputPointsFolder)
        os.makedirs(outputTripsFolder)
        os.makedirs(outputLocationsFolder)
        os.makedirs(outputDayActivitiesFolder)
        os.makedirs(outputDaySequenceFolder)
        for i in namelist:
            print i
            main_function(i, dailyPatternInterval, startHour, outputPointsFolder, outputTripsFolder,
                          outputLocationsFolder, outputDayActivitiesFolder, outputDaySequenceFolder)


def main_createLocationSequence(namelist, outputFolder, locationTimeInterval, startHour):
    # Create location sequences from trips
    if not os.path.isdir(outputFolder):
        os.makedirs(outputFolder)
    for i in namelist:
        filename = os.path.basename(i)
        fileBaseName = filename.split('_')[0]
        outputname = outputFolder + '/' + fileBaseName + '_loactionSequence.csv'
        locationSequenceCreate(i, outputname, locationTimeInterval, startHour)


def convertLocationSeqStrToList(seqStr):
    seqStr = seqStr.split('[')[1]
    seqStr = seqStr.split(']')[0]
    seqList = seqStr.split(',')
    seqListT = [int(float(x)) for x in seqList]
    return seqListT


def readLocationSeqListFromCSV(name):
    df = pd.read_csv(name, names=['date', 'seq'], header=None)
    locationSeqList = []
    for i in df.index.values:
        locationSeqList.append(convertLocationSeqStrToList(df.loc[i, 'seq']))
    return locationSeqList


def plotLocationSequence(locationSeqList, outputFileName):
    levels = [-100000000, -99.5, -98.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 1000000000]
    colors = ['black', 'green', 'purple', 'yellow', 'red', '#E3F2FD', '#2962FF', '#BBDEFB', '#2979FF', '#90CAF9',
              '#448AFF', '#64B5F6', '#0D47A1', '#42A5F5']
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
    fig, ax = plt.subplots()
    image = locationSeqList
    ax.imshow(image, cmap=cmap, norm=norm, interpolation='none')
    # ax.imshow(image, cmap=plt.cm.gray, interpolation='none')
    # ax.set_title('dropped spines')
    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    start, end = ax.get_xlim()
    step = 24
    ax.xaxis.set_ticks(np.arange(start, end + step, step))
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ['2:00AM', '4:00AM', '6:00AM', '8:00AM', '10:00AM', '12:00PM', '2:00PM', '4:00PM', '6:00PM', '8:00PM',
              '10:00PM', '12:00AM', '2:00AM']
    # labels = ['2:00AM','6:00AM','10:00AM','2:00PM','6:00PM','10:00PM','2:00AM']
    ax.set_xticklabels(labels)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Time of day')
    ax.set_ylabel('Observed days')
    # plt.show()
    fig.set_size_inches(16, 12)
    fig.set_dpi(100)
    fig.savefig(outputFileName, transparent=True, bbox_inches='tight', pad_inches=0)


def plotASingleFile(i, locationPlotFolderName):
    if not os.path.isdir(locationPlotFolderName):
        os.makedirs(locationPlotFolderName)
    filename = os.path.basename(i)
    print filename
    part1 = filename.split('_')[0]
    outputname = locationPlotFolderName + '/' + part1 + '_locationSequencePlot.png'
    plotLocationSequence(readLocationSeqListFromCSV(i), outputname)


def main_plotLocationSequence(locationSequencelist, locationPlotFolderName):
    for i in locationSequencelist:
        plotASingleFile(i, locationPlotFolderName)


def locationListCompare(trueList, predList):
    n = min(len(trueList), len(predList))
    resList = []
    for i in range(n):
        resList.append(trueList[i] == predList[i])
    return resList


def locationPredictionPerformanceForAllMethod(predictionFile, locationSeqFile,
                                              method=['predictedDay1', 'predictedDay2', 'predictedDay5',
                                                      'predictedDay10', 'bestDay1']):
    df = pd.read_csv(predictionFile, index_col=0)
    locationSeqDF = pd.read_csv(locationSeqFile, names=['date', 'seq'], header=None)
    locationSeqDF['seqList'] = None
    for i in locationSeqDF.index.values:
        locationSeqDF.loc[i, 'seqList'] = convertLocationSeqStrToList(locationSeqDF.loc[i, 'seq'])
    locationSeqDF = locationSeqDF.set_index(locationSeqDF['date'])
    del locationSeqDF['date']
    del locationSeqDF['seq']
    df = df.dropna()
    resultList = []
    for i in method:
        resultList.append(locationPredictionPerformance(df, locationSeqDF, i))
    return resultList


def locationPredictionPerformance(df, locationSeqDF, method):
    compareList = []
    for i in df.index.values:
        predict10 = df.loc[i, method]
        trueList = locationSeqDF.loc[i, 'seqList']
        predList = locationSeqDF.loc[predict10, 'seqList']
        compareList.append(locationListCompare(trueList, predList))
    df['compare'] = compareList
    length = len(compareList[0])
    resList = []
    for i in range(length):
        tempList = []
        for k in df.index.values:
            tempList.append(df.loc[k, 'compare'][i])
        resList.append(np.mean(tempList))
    return resList


def location_summary(resDF, locationSummaryOutputName):
    df = pd.DataFrame()
    length = len(resDF.iloc[0, 0])
    methodList = ['method1', 'method2', 'method5', 'method10', 'bestMethod']
    for method in methodList:
        resList = []
        for i in range(length):
            tempList = []
            for k in resDF.index.values:
                tempList.append(resDF.loc[k, method][i])
            resList.append(np.mean(tempList))
        df[method] = resList
    df.to_csv(locationSummaryOutputName)
    return df


def main_locationPredictionPerformance(predictionList, locationSeqFolder, outputName, locationSummaryOutputName,
                                       method=['predictDay1', 'predictDay2', 'predictDay5', 'predictDay10',
                                               'bestDay1']):
    resDF = pd.DataFrame()
    resDF['method1'] = [[1, 2]]
    resDF['method2'] = [[1, 2]]
    resDF['method5'] = [[1, 2]]
    resDF['method10'] = [[1, 2]]
    resDF['bestMethod'] = [[1, 2]]
    k = 0
    for predictionFile in predictionList:
        basePred = os.path.basename(predictionFile)
        part1 = basePred.split('_')[0]
        print k, ':', part1
        k += 1
        locationSeqFile = locationSeqFolder + '/' + part1 + '_trips.csv'
        [l1, l2, l3, l4, l5] = locationPredictionPerformanceForAllMethod(predictionFile, locationSeqFile,
                                                                         method=['predictDay1', 'predictDay2',
                                                                                 'predictDay5', 'predictDay10',
                                                                                 'bestDay1'])
        resDF.loc[part1, 'method1'] = l1
        resDF.loc[part1, 'method2'] = l2
        resDF.loc[part1, 'method5'] = l3
        resDF.loc[part1, 'method10'] = l4
        resDF.loc[part1, 'bestMethod'] = l5
    resDF = resDF.drop(resDF.index[0])
    resDF.to_csv(outputName)
    return location_summary(resDF, locationSummaryOutputName)


def main_locationPredictionPerformance_considerOrigin(predictionList, locationSeqFolder, outputName,
                                                      locationSummaryOutputName,
                                                      method=['predictDay1', 'predictDay2', 'predictDay5',
                                                              'predictDay10', 'bestDay1']):
    '''
    Make some changes to the begining of the day, considering that people tend to stay at the same location for short period
    '''
    resDF = pd.DataFrame()
    resDF['method1'] = [[1, 2]]
    resDF['method2'] = [[1, 2]]
    resDF['method5'] = [[1, 2]]
    resDF['method10'] = [[1, 2]]
    resDF['bestMethod'] = [[1, 2]]
    k = 0
    for predictionFile in predictionList:
        basePred = os.path.basename(predictionFile)
        part1 = basePred.split('_')[0]
        print k, ':', part1
        k += 1
        locationSeqFile = locationSeqFolder + '/' + part1 + '_trips.csv'
        [l1, l2, l3, l4, l5] = locationPredictionPerformanceForAllMethod_considerOrigin(predictionFile, locationSeqFile,
                                                                                        method=['predictDay1',
                                                                                                'predictDay2',
                                                                                                'predictDay5',
                                                                                                'predictDay10',
                                                                                                'bestDay1'])
        resDF.loc[part1, 'method1'] = l1
        resDF.loc[part1, 'method2'] = l2
        resDF.loc[part1, 'method5'] = l3
        resDF.loc[part1, 'method10'] = l4
        resDF.loc[part1, 'bestMethod'] = l5
    resDF = resDF.drop(resDF.index[0])
    resDF.to_csv(outputName)
    return location_summary(resDF, locationSummaryOutputName)


def locationPredictionPerformanceForAllMethod_considerOrigin(predictionFile, locationSeqFile,
                                                             method=['predictedDay1', 'predictedDay2', 'predictedDay5',
                                                                     'predictedDay10', 'bestDay1']):
    df = pd.read_csv(predictionFile, index_col=0)
    locationSeqDF = pd.read_csv(locationSeqFile, names=['date', 'seq'], header=None)
    locationSeqDF['seqList'] = None
    for i in locationSeqDF.index.values:
        print i
        locationSeqDF.loc[i, 'seqList'] = convertLocationSeqStrToList(locationSeqDF.loc[i, 'seq'])
    locationSeqDF = locationSeqDF.set_index(locationSeqDF['date'])
    del locationSeqDF['date']
    del locationSeqDF['seq']
    df = df.dropna()
    resultList = []
    for i in method:
        resultList.append(locationPredictionPerformance_considerOrigin(df, locationSeqDF, i))
    return resultList


def locationPredictionPerformance_considerOrigin(df, locationSeqDF, method):
    compareList = []
    for i in df.index.values:
        predict10 = df.loc[i, method]
        trueList = locationSeqDF.loc[i, 'seqList']
        predList = locationSeqDF.loc[predict10, 'seqList']
        compareList.append(locationListCompare_considerOrigin(trueList, predList))
    df['compare'] = compareList
    length = len(compareList[0])
    resList = []
    for i in range(length):
        tempList = []
        for k in df.index.values:
            tempList.append(df.loc[k, 'compare'][i])
        resList.append(np.mean(tempList))
    return resList


def locationListCompare_considerOrigin(trueList, predList):
    n = min(len(trueList), len(predList))
    predList_O = adjustLocationList_considerOrigin(predList, trueList[0])
    if trueList[0] != predList[0] and trueList[0] != -99 and predList[0] != -99:
        print 'true:', trueList
        print 'predict:', predList
        print 'adjust:', predList_O
    resList = []
    for i in range(n):
        resList.append(trueList[i] == predList_O[i])
    return resList


def adjustLocationList_considerOrigin(predList, origin):
    predListOrigin = predList[0]
    resList = predList
    if (origin != -99 and predListOrigin != -99):
        flag = 1
        resList = []
        for i in range(len(predList)):
            if flag:
                if predList[i] == predListOrigin:
                    resList.append(origin)
                else:
                    resList.append(predList[i])
                    flag = 0
            else:
                resList.append(predList[i])
    return resList


def plot_locationPrediction(resDF, outputFileName):
    resDF.columns = ['YD', 'WD', 'NN', 'GNN', 'BP']
    ax = resDF.plot(kind='line', color=['red', 'orange', 'green', 'black', 'blue'])
    start, end = ax.get_xlim()
    step = 24
    ax.xaxis.set_ticks(np.arange(start, end + step, step))
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ['2:00AM', '4:00AM', '6:00AM', '8:00AM', '10:00AM', '12:00PM', '2:00PM', '4:00PM', '6:00PM', '8:00PM',
              '10:00PM', '12:00AM', '2:00AM']
    ax.set_xticklabels(labels)
    ax.set_xlabel('Time of day')
    ax.set_ylabel('Prediction Accuracy')
    fig = ax.get_figure()
    fig.set_size_inches(16, 12)
    fig.set_dpi(100)
    fig.savefig(outputFileName, transparent=True, bbox_inches='tight', pad_inches=0)


def convertLatLngToZip(latLng):
    geolocator = GoogleV3()
    addressList = geolocator.reverse(latLng)
    if len(addressList) > 0:
        address = addressList[0][0]
        # u'5000-5028 Solomons Island Rd, Lothian, MD 20711, USA'
        addressParts = address.split(', ')
        addressPart = addressParts[len(addressParts) - 2]
        # MD 20711
        numberPart = [int(s) for s in addressPart.split() if s.isdigit()]
        return numberPart


def trainingStartingDate(startingDate, trainingSize, maxTrainingDate=30):
    if trainingSize - maxTrainingDate < startingDate:
        return startingDate
    else:
        return trainingSize - maxTrainingDate


def numRainingDays(precipitationList, length):
    temp = precipitationList[:length]
    rainList = [x for x in temp if x > 0]
    return len(rainList)


def timePeriod_locationTimeDF(time, timeInterval_locationTime):
    '''
    Find time period for a time given time interval used for locationTime dataframe
    '''
    # dayStartTime = eastern.localize( datetime(int(time.year), int(time.month), int(time.day),0))
    # print 'time',time
    dayStartTime = datetime(int(time.year), int(time.month), int(time.day), 0)
    temp_delta = time - dayStartTime
    timePeriod = ceil(temp_delta.total_seconds() / (timeInterval_locationTime * 60 * 1.0))

    currentStartTime = dayStartTime + timedelta(seconds=timePeriod * 60 * timeInterval_locationTime)
    return [currentStartTime, nextPeriod(timePeriod, timeInterval_locationTime)]


def nextPeriod(timePeriod, timeInterval_locationTime):
    n = 24 * 60 / timeInterval_locationTime
    period = (timePeriod + 1) % n
    if period == 0:
        period = n
    return period


def nextTime(time, timeInterval_locationTime):
    [nextTime, period] = timePeriod_locationTimeDF(time, timeInterval_locationTime)
    if nextTime == time:
        nextTime = time + timedelta(minutes=timeInterval_locationTime)
    return nextTime


def addFirstRecord_locationTimeDF(locDF, tripStartTime, tripEndTime, timeInterval_locationTime, trip_ID,
                                  precipitationDict):
    '''
    Create frist record for locationTimeDF
    '''
    [currentTime, currentPeriod] = timePeriod_locationTimeDF(tripStartTime, timeInterval_locationTime)
    # print tripStartTime,currentTime
    locationList = []
    timeList = []
    if currentTime > tripEndTime:
        # print currentTime, locationList
        return [locDF, currentTime, currentPeriod, locationList, timeList]
    else:
        return processTripRecord_locationTimeDF(locDF, tripStartTime, tripEndTime, timeInterval_locationTime, trip_ID,
                                                currentTime, currentPeriod, locationList, timeList, precipitationDict)


def addOtherRecord_locationTimeDF(locDF, currentTime, currentPeriod, locationList, timeList, stayStartTime,
                                  tripStartTime, tripEndTime, timeInterval_locationTime, trip_ID, location_ID,
                                  precipitationDict):
    '''
    Create other record for locationTimeDF
    '''
    # print 'start stay'
    [locDF, currentTime, currentPeriod, locationList, timeList] = processTripRecord_locationTimeDF(locDF, stayStartTime,
                                                                                                   tripStartTime,
                                                                                                   timeInterval_locationTime,
                                                                                                   location_ID,
                                                                                                   currentTime,
                                                                                                   currentPeriod,
                                                                                                   locationList,
                                                                                                   timeList,
                                                                                                   precipitationDict)
    # print 'finish stay'
    [locDF, currentTime, currentPeriod, locationList, timeList] = processTripRecord_locationTimeDF(locDF, tripStartTime,
                                                                                                   tripEndTime,
                                                                                                   timeInterval_locationTime,
                                                                                                   trip_ID, currentTime,
                                                                                                   currentPeriod,
                                                                                                   locationList,
                                                                                                   timeList,
                                                                                                   precipitationDict)
    # print 'finish trip'
    return [locDF, currentTime, currentPeriod, locationList, timeList]


def findOneLocInList(locList):
    '''
    In 30min, if 10min home (0), 10min work (1), 10min travel (-99)
    '''
    tempList = [i for i in locList if i >= 0]
    if len(tempList) > 0:
        return min(tempList)
    else:
        return max(locList)


def determineID(locationList, timeList):
    '''
    Give a list of locations and a list of time, find the largest time and corresponding location ID
    '''
    m = max(timeList)
    indexList = [i for i, j in enumerate(timeList) if j == m]
    locList = [locationList[i] for i in indexList]
    if len(locList) == 1:
        return locList[0]
    else:
        return findOneLocInList(locList)


def processTripRecord_locationTimeDF(locDF, startTime, endTime, timeInterval_locationTime, ID, currentTime,
                                     currentPeriod, locationList, timeList, precipitationDict):
    tempTime = nextTime(currentTime, timeInterval_locationTime)
    tempPeriod = nextPeriod(currentPeriod, timeInterval_locationTime)
    if tempTime <= endTime:
        locationList.append(ID)
        timeList.append((tempTime - currentTime).total_seconds())
        addID = determineID(locationList, timeList)
        locationList = []
        timeList = []
        # print tempTime.date()
        if tempTime.date() in precipitationDict:
            locDF.loc[len(locDF)] = [tempTime.date(), tempTime, tempTime.weekday(), currentPeriod,
                                     precipitationDict[tempTime.date()], addID]
        currentTime = tempTime
        currentPeriod = tempPeriod
        # print currentTime, locationList
        return processTripRecord_locationTimeDF(locDF, startTime, endTime, timeInterval_locationTime, ID, currentTime,
                                                currentPeriod, locationList, timeList, precipitationDict)
    else:
        # print currentTime, locationList
        locationList.append(ID)
        timeList.append((endTime - currentTime).total_seconds())
        currentTime = endTime
        # print currentTime
        return [locDF, currentTime, currentPeriod, locationList, timeList]


def makeDict(keyList, valueList):
    dictionary = {}
    for i in range(len(keyList)):
        key = keyList[i]
        value = valueList[i]
        # print key,value
        dictionary[key] = value
    return dictionary


def readPrecipitationDictFromCSV(name):
    dayIndex = []
    dayPrecipitation = []
    df = pd.read_csv(name, index_col=0)
    for i in df.index.values:
        dayIndex.append(try_parsing_date(i))
        dayPrecipitation.append(getNumberPrecipitation(df.loc[i, 'PrecipitationIn']))
    return makeDict(dayIndex, dayPrecipitation)


def createLocationTimeDF(tripDFName, weatherPath, outPath, timeInterval_locationTime):
    basename = os.path.basename(tripDFName)
    print basename
    weatherFileName = weatherPath + '/' + basename.split('_')[0] + '_weather.csv'
    outFileName = outPath + '/' + basename.split('_')[0] + '_locationTimeDF.csv'
    if os.path.isfile(outFileName):
        print 'skip'
        # If we have already created the file, then we don't need to go through the calculation again
        return
    print 'processing'
    precipitationDict = readPrecipitationDictFromCSV(weatherFileName)
    tripDF = readTripsFromCSV(tripDFName)
    timeInterval_locationTime = int(timeInterval_locationTime)
    locDF = pd.DataFrame()
    locDF['Day'] = None
    locDF['Time'] = None
    locDF['DayOfWeek'] = None
    locDF['TimePeriod'] = None
    locDF['DayPrecipitation'] = None
    locDF['locID'] = None
    firstFlag = 0
    unobserved_LocID = -1
    trip_ID = -99
    leftSec = 0
    # Loop through each record in the tripDF
    for i in tripDF.index.values:
        # print i
        # get the location ID
        location_ID = tripDF.loc[i, 'locationID_O']
        tripStartTime = tripDF.loc[i, 'start_time']
        tripEndTime = tripDF.loc[i, 'end_time']
        if pd.isnull(location_ID):
            location_ID = unobserved_LocID
            unobserved_LocID = unobserved_LocID - 1
        if firstFlag == 0:
            # If it is the first trip of the people
            # 1. Figure out the corresponding start date and time
            [locDF, currentTime, currentPeriod, locationList, timeList] = addFirstRecord_locationTimeDF(locDF,
                                                                                                        tripStartTime,
                                                                                                        tripEndTime,
                                                                                                        timeInterval_locationTime,
                                                                                                        trip_ID,
                                                                                                        precipitationDict)
            # print currentTime, locationList
            # print locDF
            firstFlag = 1
            stayStartTime = tripEndTime
        else:
            # if it is not the first record
            # print 'DayStartTime:',dayStartTime,'currentTime',currentTime, 'tripStartTime:',tripStartTime,'tripEndTime',tripEndTime
            [locDF, currentTime, currentPeriod, locationList, timeList] = addOtherRecord_locationTimeDF(locDF,
                                                                                                        currentTime,
                                                                                                        currentPeriod,
                                                                                                        locationList,
                                                                                                        timeList,
                                                                                                        stayStartTime,
                                                                                                        tripStartTime,
                                                                                                        tripEndTime,
                                                                                                        timeInterval_locationTime,
                                                                                                        trip_ID,
                                                                                                        location_ID,
                                                                                                        precipitationDict)
            stayStartTime = tripEndTime
            # print currentTime,locationList
            # print locDF
    if not os.path.isdir(outPath):
        os.makedirs(outPath)
    locDF.to_csv(outFileName)


def findTrainingSizeLocation(n, trainingSizeList):
    lessList = [x for x in trainingSizeList if x <= n]
    return len(lessList) - 1


def determineSequenceMoveOrNot(seqList):
    # Determine the day sequence is stationary or not, if the people do move, return 1, otherwise, return 0
    setT = set(seqList) - set([-100])
    return len(setT) > 1


def numMovingDays(daySequence):
    # Calculate the number of moving days for the people
    counter = 0
    for i in range(len(daySequence)):
        if determineSequenceMoveOrNot(daySequence[i]):
            counter += 1
    return counter


def determineUnobservedDay(seqList):
    # Determine if there is any missing data within the day, if there is missing data, return 1, otherwise return 0
    listT = list(set(seqList))
    for i in listT:
        if i == -100:
            return 1
    return 0


def numMissingDays(daySequence):
    # Calculate the number of moving days for the people
    counter = 0
    for i in range(len(daySequence)):
        if determineUnobservedDay(daySequence[i]):
            counter += 1
    return counter


def updateWeatherDailySequenceWithNewDailySequence(sequenceNameList, outputDaySequenceFolder):
    # When I calculated new daily sequence files, I need to update the daily sequence file with weather information since weather information involves some manual work
    for weatherFile in sequenceNameList:
        fileName = os.path.basename(weatherFile).split('_')[0]
        print fileName
        nWeatherFile = outputDaySequenceFolder + '/' + fileName + '_daySequence.csv'
        df1 = pd.read_csv(weatherFile, index_col=0)
        df2 = pd.read_csv(nWeatherFile, names=['date', 'seq'], index_col=0)
        if len(df1) != len(df2):
            print 'error'
        for i in range(len(df1)):
            df1.iloc[i, 0] = df2.iloc[i, 0]
        # df1['seq'] = df2['seq']
        df1.to_csv(weatherFile)


def getCleanLocationSet(o_list):
    EXCLUDE_LIST = [-99, -100]
    r_list = [x for x in o_list if x not in EXCLUDE_LIST]
    r_list.sort()
    return r_list


def getLocationSequence(o_list):
    EXCLUDE_LIST = [-99, -100]
    o_list = [x for x in o_list if x not in EXCLUDE_LIST]
    r_list = []
    for i in range(len(o_list)):
        if i == 0:
            r_list.append(o_list[i])
        else:
            if o_list[i] != o_list[i - 1]:
                r_list.append(o_list[i])
    return r_list


def getRecurrentLocationSet(o_list, reccurent_list):
    r_list = [x for x in o_list if x in reccurent_list]
    r_list.sort()
    return r_list


def setOfLocations(sequenceNameList, outputLocationSetFolder):
    if not os.path.isdir(outputLocationSetFolder):
        os.makedirs(outputLocationSetFolder)
    for locationFile in sequenceNameList:
        fileName = os.path.basename(locationFile).split('_')[0]
        print fileName
        outfilename = outputLocationSetFolder + '/' + fileName + '_locationSet.csv'
        df = pd.read_csv(locationFile, names=['date', 'seq'], index_col=0)
        allList = []
        df['reccurentSet'] = None
        df['locationSet'] = None
        df['locationSequence'] = None
        df['weekday'] = None
        df['missingData'] = None
        for i in df.index.values:
            df.loc[i, 'locationSet'] = getCleanLocationSet(list(set(ast.literal_eval(df.loc[i, 'seq']))))
            df.loc[i, 'locationSequence'] = getLocationSequence(ast.literal_eval(df.loc[i, 'seq']))
            df.loc[i, 'weekday'] = try_parsing_date(i).weekday()
            if -100 in set(ast.literal_eval(df.loc[i, 'seq'])):
                df.loc[i, 'missingData'] = 1
            else:
                df.loc[i, 'missingData'] = 0
            allList = allList + df.loc[i, 'locationSet']
        allSet = list(set(allList))
        frequencyList = []
        for i in allSet:
            frequencyList.append(0)
        for i in range(len(allSet)):
            locationID = allSet[i]
            df[locationID] = None
            counter = 0
            for k in df.index.values:
                if locationID in df.loc[k, 'locationSet']:
                    df.loc[k, locationID] = 1
                    counter += 1
                else:
                    df.loc[k, locationID] = 0
            frequencyList[i] = counter
        recurrentList = [allSet[i] for i in range(len(allSet)) if frequencyList[i] > 1]
        for i in df.index.values:
            df.loc[i, 'reccurentSet'] = getRecurrentLocationSet(list(set(ast.literal_eval(df.loc[i, 'seq']))),
                                                                recurrentList)
        df.to_csv(outfilename)


def findStudyArea(tripList):
    minLat = 1000
    maxLat = -1000
    minLng = 1000
    maxLng = -1000
    for tripDFName in tripList:
        tripDF = readTripsFromCSV(tripDFName)
        for i in tripDF.index.values:
            minLat = min(minLat, tripDF.loc[i, 'lat_O'], tripDF.loc[i, 'lat_D'])
            maxLat = max(maxLat, tripDF.loc[i, 'lat_O'], tripDF.loc[i, 'lat_D'])
            minLng = min(minLng, tripDF.loc[i, 'long_O'], tripDF.loc[i, 'long_D'])
            maxLng = max(maxLng, tripDF.loc[i, 'long_O'], tripDF.loc[i, 'long_D'])
        print tripDFName, minLat, maxLat, minLng, maxLng
    return [minLat, maxLat, minLng, maxLng]


def locationToGridCenter(lat, lng):
    lat_center = round((lat - MIN_LAT) / GRID_LENGTH_LAT) * GRID_LENGTH_LAT + MIN_LAT
    lng_center = round((lng - MIN_LNG) / GRID_LENGTH_LNG) * GRID_LENGTH_LNG + MIN_LNG
    return [lat_center, lng_center]


def locationToZoneNumber(lat, lng):
    num_lat = int(round((lat - MIN_LAT) / GRID_LENGTH_LAT))
    num_lng = int(round((lng - MIN_LNG) / GRID_LENGTH_LNG))
    # print num_lat * ZONE + num_lng
    return num_lat * ZONE + num_lng


def zoneNumberToGridCenter(num):
    num_lat = int(num / ZONE)
    num_lng = num % ZONE
    lat_center = num_lat * GRID_LENGTH_LAT + MIN_LAT
    lng_center = num_lng * GRID_LENGTH_LNG + MIN_LNG
    return [lat_center, lng_center]


def most_common(lst):
    return max(set(lst), key=lst.count)


def determineIfBelongToStayRegion(lat, lng, zone):
    [lat_center, lng_center] = zoneNumberToGridCenter(zone)
    if lat >= lat_center - 1.5 * GRID_LENGTH_LAT and lat <= lat_center + 1.5 * GRID_LENGTH_LAT and lng >= lng_center - 1.5 * GRID_LENGTH_LNG and lng <= lng_center + 1.5 * GRID_LENGTH_LNG:
        return True
    else:
        return False


def getDestinationDurationUpperBound(tripIndex, tripDF):
    if tripDF.loc[tripIndex, 'is_duration_D_accurate'] == 1:
        return tripDF.loc[tripIndex, 'duration_D']
    else:
        return tripDF.loc[tripIndex, 'duration_D_upperbound']


def findStayRegion(df):
    df['zone_D'] = None
    df['regionID_O'] = None
    df['regionID'] = None
    locationDF = pd.DataFrame(np.nan, index=[0],
                              columns=['zone', 'lat', 'long', 'num', 'total_time', 'avg_time', 'trip_index', 'num_weekday', 'total_time_weekday',
                                       'avg_time_weekday', 'trip_index_weekday', 'num_weekend',
                                       'total_time_weekend', 'avg_time_weekend', 'trip_index_weekend'])
    locationIndex = 0
    for i in df.index.values:
        df.loc[i, 'zone_D'] = locationToZoneNumber(df.loc[i, 'lat_D_accurate'], df.loc[i, 'long_D_accurate'])
        df.loc[i, 'regionID'] = -1
    stopFlag = 0
    while (stopFlag == 0):
        zoneList = []
        for i in df.index.values:
            if not pd.isnull(df.loc[i, 'regionID']):
                if df.loc[i, 'regionID'] <= 0:
                    zoneList.append(df.loc[i, 'zone_D'])
        commonZone = most_common(zoneList)
        # print 'Common zone is ', commonZone
        totalNum = 0
        weekdayNum = 0
        weekendNum = 0
        latSum = 0
        lngSum = 0
        totalDuration = timedelta(minutes=0)
        weekdayDuration = timedelta(minutes=0)
        weekendDuration = timedelta(minutes=0)
        totalDuration_upperbound = timedelta(minutes=0)
        weekdayDuration_upperbound = timedelta(minutes=0)
        weekendDuration_upperbound = timedelta(minutes=0)
        totalString = ""
        weekdayString = ""
        weekendString = ""
        for i in df.index.values:
            if not pd.isnull(df.loc[i, 'regionID']):
                if df.loc[i, 'regionID'] <= 0 and determineIfBelongToStayRegion(df.loc[i, 'lat_D_accurate'],
                                                                                df.loc[i, 'long_D_accurate'],
                                                                                commonZone):
                    df.loc[i, 'regionID'] = commonZone
                    totalNum += 1
                    latSum = latSum + df.loc[i, 'lat_D_accurate']
                    lngSum = lngSum + df.loc[i, 'long_D_accurate']
                    totalDuration = totalDuration + df.loc[i, 'duration_D']
                    # totalDuration_upperbound = totalDuration_upperbound + getDestinationDurationUpperBound(i,df)
                    totalString = totalString + "*" + str(int(i))
                    if df.loc[i, 'weekday'] < 5:
                        weekdayNum += 1
                        weekdayDuration = weekdayDuration + df.loc[i, 'duration_D']
                        # weekdayDuration_upperbound = weekdayDuration_upperbound + getDestinationDurationUpperBound(i,df)
                        weekdayString = weekdayString + "*" + str(int(i))
                    else:
                        weekendNum += 1
                        weekendDuration = weekendDuration + df.loc[i, 'duration_D']
                        # weekendDuration_upperbound = weekendDuration_upperbound + getDestinationDurationUpperBound(i,df)
                        weekendString = weekendString + "*" + str(int(i))
        locationDF.loc[locationIndex, 'zone'] = commonZone
        locationDF.loc[locationIndex, 'lat'] = latSum / totalNum
        locationDF.loc[locationIndex, 'long'] = lngSum / totalNum
        locationDF.loc[locationIndex, 'num'] = totalNum
        locationDF.loc[locationIndex, 'total_time'] = totalDuration
        # locationDF.loc[locationIndex,'total_time_upperbound'] = totalDuration_upperbound
        if totalNum == 0:
            locationDF.loc[locationIndex, 'avg_time'] = timedelta(minutes=0)
            # locationDF.loc[locationIndex,'avg_time_upperbound'] = timedelta(minutes=0)
        else:
            locationDF.loc[locationIndex, 'avg_time'] = totalDuration / totalNum
            # locationDF.loc[locationIndex,'avg_time_upperbound'] = totalDuration_upperbound/totalNum
        locationDF.loc[locationIndex, 'trip_index'] = totalString
        locationDF.loc[locationIndex, 'num_weekday'] = weekdayNum
        locationDF.loc[locationIndex, 'total_time_weekday'] = weekdayDuration
        # locationDF.loc[locationIndex,'total_time_weekday_upperbound'] = weekdayDuration_upperbound
        if weekdayNum == 0:
            locationDF.loc[locationIndex, 'avg_time_weekday'] = timedelta(minutes=0)
            # locationDF.loc[locationIndex,'avg_time_weekday_upperbound'] = timedelta(minutes=0)
        else:
            locationDF.loc[locationIndex, 'avg_time_weekday'] = weekdayDuration / weekdayNum
            # locationDF.loc[locationIndex,'avg_time_weekday_upperbound'] = weekdayDuration_upperbound/weekdayNum
        locationDF.loc[locationIndex, 'trip_index_weekday'] = weekdayString
        locationDF.loc[locationIndex, 'num_weekend'] = weekendNum
        locationDF.loc[locationIndex, 'total_time_weekend'] = weekendDuration
        # locationDF.loc[locationIndex,'total_time_weekend_upperbound'] = weekendDuration_upperbound
        if weekendNum == 0:
            locationDF.loc[locationIndex, 'avg_time_weekend'] = timedelta(minutes=0)
            # locationDF.loc[locationIndex,'avg_time_weekend_upperbound'] = timedelta(minutes=0)
        else:
            locationDF.loc[locationIndex, 'avg_time_weekend'] = weekendDuration / weekendNum
            # locationDF.loc[locationIndex,'avg_time_weekend_upperbound'] = weekendDuration_upperbound/weekendNum
        locationDF.loc[locationIndex, 'trip_index_weekend'] = weekendString
        locationIndex += 1
        flag = 0
        for i in df.index.values:
            if not pd.isnull(df.loc[i, 'regionID']):
                flag = min(flag, df.loc[i, 'regionID'])
        if flag > -1:
            stopFlag = 1
    for i in df.index.values:
        if i > 0:
            df.loc[i, 'regionID_O'] = df.loc[i - 1, 'regionID']
    return [df, locationDF]


def main_function_createLocationSequenceTable(inputfilename, dailyPatternInterval, startHour, writeFolderName1,
                                              writeFolderName2, writeFolderName3, writeFolderName4, writeFolderName5):
    filename = os.path.basename(inputfilename).split('_')[0]
    print filename
    outputName1 = writeFolderName1 + '/' + filename + '_points.csv'
    outputName2 = writeFolderName2 + '/' + filename + '_trips.csv'
    outputName3 = writeFolderName3 + '/' + filename + '_locations.csv'
    outputName4 = writeFolderName4 + '/' + filename + '_daySummary.csv'
    # outputName5 = writeFolderName5 + '/' + filename + '_daySequence.csv'
    # i = 'C:/Users/LIANG/OneDrive/Travel Pattern Prediction/Processed_Data/GPS_trips_processed/2-JodieAKulpaEddy_trips.csv'
    # out = 'C:/Users/LIANG/OneDrive/Travel Pattern Prediction/test.csv'
    # out1 = 'C:/Users/LIANG/OneDrive/Travel Pattern Prediction/locationTest.csv'
    df = readTripsFromCSV(outputName2)
    # print df.head()
    print 'findStayRegion'
    [df, locationDF] = findStayRegion(df)
    print 'matchPointToLocationID_stayRegion'
    df = matchPointToLocationID_stayRegion(df, locationDF)
    print 'filterTrip_deleteCircleTrip_stayRegion'
    df = filterTrip_deleteCircleTrip_stayRegion(df)
    # print 'updateDurationUpperBound'
    # df = updateDurationUpperBound(df)
    locationDF = updateLocationDF_stayRegion(df, locationDF)
    locationDF = locationDF.sort_values(['total_time'], ascending=[False])
    locationDF = locationDF.reset_index(drop=True)
    locationDF['rank_totalTime'] = locationDF['total_time'].rank(ascending=False)
    numList = list(locationDF.index.values)
    zoneList = list(locationDF['zone'])
    zoneDict = makeDict(zoneList, numList)
    df['locationID_O'] = None
    df['locationID_D'] = None
    for k in df.index.values:
        if not pd.isnull(df.loc[k, 'regionID_O']):
            df.loc[k, 'locationID_O'] = zoneDict[df.loc[k, 'regionID_O']]
        if not pd.isnull(df.loc[k, 'regionID']):
            df.loc[k, 'locationID_D'] = zoneDict[df.loc[k, 'regionID']]
    df.to_csv(outputName2)
    locationDF.to_csv(outputName3)
    print 'daySequenceCreate'
    # DaySequenceSeries = daySequenceCreate(df, outputName5, dailyPatternInterval, startHour)
    return

def main_function_createDaySequence(inputfilename, dailyPatternInterval, startHour, writeFolderName5):
    filename = os.path.basename(inputfilename).split('_')[0]
    print filename
    df = readTripsFromCSV(inputfilename)
    outputName5 = writeFolderName5 + '/' + filename + '_daySequence.csv'
    daySequenceCreate(df, outputName5, dailyPatternInterval, startHour)
    return

def matchPointToLocationID_stayRegion(tripDF, locationDF):
    for i in tripDF.index.values:
        # Fill the locationID_O
        if pd.isnull(tripDF.loc[i, 'regionID_O']):
            print 'the trip ', i, 'has no region ID'
            if not pd.isnull(tripDF.loc[i, 'lat_PreviousD_accurate']):
                lat_PreviousD = tripDF.loc[i, 'lat_PreviousD_accurate']
                long_PreviousD = tripDF.loc[i, 'long_PreviousD_accurate']
            else:
                lat_PreviousD = tripDF.loc[i, 'lat_PreviousD']
                long_PreviousD = tripDF.loc[i, 'long_PreviousD']
            if not pd.isnull(tripDF.loc[i, 'lat_O_accurate']):
                lat_O = tripDF.loc[i, 'lat_O_accurate']
                long_O = tripDF.loc[i, 'long_O_accurate']
            else:
                lat_O = tripDF.loc[i, 'lat_O']
                long_O = tripDF.loc[i, 'long_O']
            zoneID = findNearestLocationID_stayRegion(lat_PreviousD, long_PreviousD, lat_O, long_O, locationDF)
            print 'zoneID is ', zoneID
            if not pd.isnull(zoneID):
                tripDF.loc[i, 'regionID_O'] = zoneID
                print 'add id to trip ', i, 'with zone id ', zoneID
                # Fill the locationID_D
                if i > 0:
                    tripDF.loc[i - 1, 'regionID'] = zoneID
                    # if tripDF.loc[i-1,'missing_Data'] == 0:
                    #    print 'add trip ',i-1
                    #    addTripToLocationDF_stayRegion(i-1, tripDF, zoneID, locationDF)
    return tripDF


def findNearestLocationID_stayRegion(lat1, long1, lat2, long2, locationDF):
    # zoneList = list(locationDF['zone'])
    # zone1 = -1
    # zone2 = -1
    # if not pd.isnull(lat1):
    #    zone1 = locationToZoneNumber(lat1,long1)
    # if not pd.isnull(lat2):
    #    zone2 = locationToZoneNumber(lat2,long2)
    # print 'zone 1 ',zone1, 'zone 2 ',zone2
    # if zone1 in zoneList and zone2 in zoneList:
    #    zone1Index = zoneList.index(zone1)
    #    zone2Index = zoneList.index(zone2)
    #    zoneIndex = min(zone1Index, zone2Index)
    #    return zoneList[zoneIndex]
    # elif zone1 in zoneList:
    #    return zone1
    # elif zone2 in zoneList:
    #    return zone2
    # else:
    #    return None
    for i in locationDF.index.values:
        if not np.isnan(lat1):
            dist1 = haversine(long1, lat1, locationDF.loc[i, 'long'], locationDF.loc[i, 'lat'])
            if dist1 < 300:
                return locationDF.loc[i, 'zone']
        if not np.isnan(lat2):
            dist2 = haversine(long2, lat2, locationDF.loc[i, 'long'], locationDF.loc[i, 'lat'])
            if dist2 < 300:
                return locationDF.loc[i, 'zone']
    return None


def addTripToLocationDF_stayRegion(i_tripDF, tripDF, zoneID, locationDF):
    if not pd.isnull(zoneID):
        zoneList = list(locationDF['zone'])
        zoneIndex = zoneList.index(zoneID)
        locationDF.loc[zoneIndex, 'num'] += 1
        locationDF.loc[zoneIndex, 'total_time'] += tripDF.loc[i_tripDF, 'duration_D']
        # print 'Total_time:', locationDF.loc[zoneIndex, 'total_time']
        # print 'num:', locationDF.loc[zoneIndex, 'num']
        locationDF.loc[zoneIndex, 'avg_time'] = locationDF.loc[zoneIndex, 'total_time'] / int(locationDF.loc[
            zoneIndex, 'num'])
        locationDF.loc[zoneIndex, 'trip_index'] += '*'
        locationDF.loc[zoneIndex, 'trip_index'] += str(i_tripDF)
        if tripDF.loc[i_tripDF, 'weekday'] < 5:
            locationDF.loc[zoneIndex, 'num_weekday'] += 1
            locationDF.loc[zoneIndex, 'total_time_weekday'] += tripDF.loc[i_tripDF, 'duration_D']
            locationDF.loc[zoneIndex, 'avg_time_weekday'] = locationDF.loc[zoneIndex, 'total_time_weekday'] / \
                                                            int(locationDF.loc[zoneIndex, 'num_weekday'])
            locationDF.loc[zoneIndex, 'trip_index_weekday'] += '*'
            locationDF.loc[zoneIndex, 'trip_index_weekday'] += str(i_tripDF)
        else:
            locationDF.loc[zoneIndex, 'num_weekend'] += 1
            locationDF.loc[zoneIndex, 'total_time_weekend'] += tripDF.loc[i_tripDF, 'duration_D']
            locationDF.loc[zoneIndex, 'avg_time_weekend'] = locationDF.loc[zoneIndex, 'total_time_weekend'] / \
                                                            int(locationDF.loc[zoneIndex, 'num_weekend'])
            locationDF.loc[zoneIndex, 'trip_index_weekend'] += '*'
            locationDF.loc[zoneIndex, 'trip_index_weekend'] += str(i_tripDF)
    return


def updateLocationDF_stayRegion(tripDF, locationDF):
    for i in locationDF.index.values:
        locationDF.loc[i, 'num'] = 0
        locationDF.loc[i, 'total_time'] = timedelta(minutes=0)
        locationDF.loc[i, 'avg_time'] = timedelta(minutes=0)
        locationDF.loc[i, 'trip_index'] = str()
        locationDF.loc[i, 'num_weekend'] = 0
        locationDF.loc[i, 'total_time_weekend'] = timedelta(minutes=0)
        locationDF.loc[i, 'avg_time_weekend'] = timedelta(minutes=0)
        locationDF.loc[i, 'trip_index_weekend'] = str()
        locationDF.loc[i, 'num_weekday'] = 0
        locationDF.loc[i, 'total_time_weekday'] = timedelta(minutes=0)
        locationDF.loc[i, 'avg_time_weekday'] = timedelta(minutes=0)
        locationDF.loc[i, 'trip_index_weekday'] = str()
    for i in tripDF.index.values:
        if not pd.isnull(tripDF.loc[i, 'regionID']):
            addTripToLocationDF_stayRegion(i, tripDF, tripDF.loc[i, 'regionID'], locationDF)
    return locationDF


def seperateRawDataIntoIndividualData(android_data_name, out_folder_path):
    '''
    Seperate the raw GPS data into individual GPS data files, named with the mobileID
    :param android_data_name: Raw GPS data csv file exported from the Database
    :param out_folder_path: Folder the output files saved to
    :return: None
    '''
    if not os.path.isdir(out_folder_path):
        os.makedirs(out_folder_path)

    datafile = open(android_data_name, 'r')
    reader = csv.reader(datafile, delimiter=';')
    next(reader, None)
    for line in reader:
        data = line
        if data[5] != '' and data[1] != '' and data[2] != '' and data[3]!='' and data[4]!='':
            out_name = out_folder_path + '/' + data[0] + '.csv'
            out_file = open(out_name, 'a')
            out_file.write(data[0] + ';' + data[1] + ';' + data[2] + ';' + data[3] + ';' + data[4] + ';' + data[5] + '\n')
            out_file.close
    return