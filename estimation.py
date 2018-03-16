import os
import glob
import csv
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime, timedelta, date
from scipy.stats import gaussian_kde
import numpy as np
from gurobipy import *
import time
from scipy.optimize import minimize,differential_evolution
from scipy import optimize
from scikits.optimization import *
import DataProcessingFunctions_Android
reload(DataProcessingFunctions_Android)

import os
from os.path import dirname, abspath
dir_path = dirname(os.path.realpath(__file__))
parent_path = dirname(dirname(dirname(abspath(dir_path))))
print dir_path
print parent_path
os.chdir(parent_path)

START_HOUR = 2
MIN_STARTTIME = 120
MAX_STARTTIME = 1560
MIN_DURATION = 0
MAX_DURATION = 1440
dailyPatternInterval = 20

# sequenceAlignmentBio(self.daySequence[k],self.daySequence[i])
parent_path = 'D:/Projects/Liang Tang/Travel Pattern Prediction'
code_path = parent_path + '/Python_code/venv/Scripts/'
sys.path.append(code_path)
import DataProcessingFunctions_Android
from SPSA import minimizeSPSA_Liang,minimizeSPSA_Customize
from MCS import MCS

def findWorkStartTime(g):
    g = g.reset_index(drop=True)
    startTime = g.loc[0,'end_time']
    startTimeT = startTime.hour * 60 + startTime.minute + startTime.second/60.0
    return pd.Series(dict(workStartTime = startTimeT))

def findWorkEndTime(g):
    g = g.reset_index(drop=True)
    endTime = g.loc[len(g)-1,'start_time']
    endTimeT = endTime.hour * 60 + endTime.minute + endTime.second/60.0
    return pd.Series(dict(workEndTime = endTimeT))

def findHomeWorkVistTimes(g):
    return pd.Series(dict(visitTimes = len(g)))

def max_density(density, startTime,endTime):
    # Find the maximum value from kenel density
    ys = density(np.arange(startTime,endTime))
    bb = np.argmax(ys)
    return np.arange(startTime,endTime)[bb]

def findMaxProKernel(x_list,startTime,endTime):
    if len(x_list) == 1:
        return x_list[0]
    density = gaussian_kde(x_list)
    return max_density(density,startTime,endTime)

def calInitialU(ID,df_trip,start_date_train,deltaU):
    df_trip_2 = df_trip[df_trip['locationID_D']== ID]
    df_trip_2 = df_trip_2.reset_index(drop=True)
    date_2 = df_trip_2.loc[0,'end_time'].to_datetime().date()
    distance = date_2 - start_date_train
    distance = distance.days
    avg_distance = len(seq_train)/location_frequent_dict[2]
    initial_u = deltaU[ID]
    if distance < avg_distance:
        initial_u = deltaU[ID] * (avg_distance - distance)
    return initial_u

def replaceHomeWorkLocations(o_dict,ID,ID_list):
    d_dict = {}
    for i in o_dict:
        if i == ID:
            for ID_virtual in ID_list:
                d_dict[ID_virtual] = o_dict[i]
        else:
            d_dict[i] = o_dict[i]
    return d_dict

def replaceHomeWorkLinks(o_dict,ID,ID_list):
    d_dict = {}
    for OD in o_dict:
        if OD[0] == ID:
            for ID_virtual in ID_list:
                OD_pair = (ID_virtual,OD[1])
                d_dict[OD_pair] = o_dict[OD]
        elif OD[1] == ID:
            for ID_virtual in ID_list:
                OD_pair = (OD[0],ID_virtual)
                d_dict[OD_pair] = o_dict[OD]
        else:
            d_dict[OD] = o_dict[OD]
    return d_dict

def daySeqRemoveTravel(o_seq):
    return [x for x in o_seq if x!=-99]

# Convert Gurobi results to day sequence
def convertGurobiSolutionToDaySeq(node, start_time, end_time):
    node_visited = [x for x in locations if node[x].X > 0]
    start_time_visited = [start_time[x].X for x in node_visited]
    end_time_visited = [end_time[x].X for x in node_visited]
    # print 'node visited:',node_visited
    # print 'start time visited', start_time_visited
    sequence = sorted(range(len(start_time_visited)), key=lambda k: start_time_visited[k])
    predict_day_seq = []
    for index in sequence:
        node_ID = node_visited[index]
        if node_ID in home_locations:
            node_ID = home_ID
        elif node_ID in work_locations:
            node_ID = work_ID
        duration_temp = end_time_visited[index] - start_time_visited[index]
        n1 = round(duration_temp * 1.0 / dailyPatternInterval)
        n1 = int(n1)
        if n1 == 0:
            n1 = 1
        for i in range(int(n1)):
            predict_day_seq.append(node_ID)
    return predict_day_seq

def updateInitialUtility(ID, initialU,previous_observed_seq, deltaU):
    if ID in previous_observed_seq:
        return deltaU[ID]
    else:
        return initialU[ID] + deltaU[ID]

def intervalAccuracy(seq_train, interval, ID):
    current_distance = 1
    correct_days= 0
    for i in range(len(seq_train)):
        # print seq_train[i]
        if current_distance >= interval and ID in seq_train[i]:
            correct_days += 1
        elif current_distance < interval and ID not in seq_train[i]:
            correct_days += 1
        # else:
        #     print seq_train.index.values[i]
        if ID in seq_train[i]:
            current_distance = 0
        current_distance += 1
    # print 'ID:',ID, 'interval:',interval,'correct days:',correct_days,'/',len(seq_train)
    return correct_days

def weekdayAccuracy(seq,nonwork_days_boolean,ID ):
    correct_days = 0
    for i in seq.index.values:
        weekday = i.weekday()
        if nonwork_days_boolean[(ID,weekday)] == 0 and ID not in seq[i]:
            correct_days += 1
        elif nonwork_days_boolean[(ID,weekday)] == 1 and ID in seq[i]:
            correct_days += 1
    return correct_days

#
# test_path = parent_path + '/App_data/test/'
# # test_file = parent_path + '/App_data/test/714d2ea9-ea8c-4b20-8988-c88e80efb86d_trips.csv'
# # test_file = parent_path + '/App_data/test/1d1d260a-3db9-45d4-8785-120f268a50a2_trips.csv'
# test_file = parent_path + '/App_data/test/3f81bff9-2613-4118-8606-450a6a568d95_trips.csv'
# fileID = os.path.basename(test_file).split('_')[0]
# print fileID
# seq_file = test_path + fileID + '_daySequence.csv'
# trip_file = test_path + fileID + '_trips.csv'
# seq = DataProcessingFunctions_Android.readDaySequenceFromCSV(seq_file)
# seq1 = seq[1:]

tripList = glob.glob('App_data/Processed_Data/GPS_trips_processed/*.csv')
DAYSEQ_PATH = 'App_data/Processed_Data/GPS_daySequence_processed/'
SUMMARY_PATH = 'App_data/Processed_Data/GPS_daySequence_processed_summary/'
if not os.path.isdir(SUMMARY_PATH):
    os.mkdir(SUMMARY_PATH)

def evaluatePersonFile(trip_file):
    fileID = os.path.basename(trip_file).split('_')[0]
    print fileID
    seq_file = DAYSEQ_PATH + fileID + '_daySequence.csv'
    seq = DataProcessingFunctions_Android.readDaySequenceFromCSV(seq_file)
    seq1 = seq[1:]
    observed_day_length = len(seq1)
    N_DAYS = 60
    if N_DAYS > observed_day_length:
        N_DAYS = int(0.8 * observed_day_length)
    print 'Training data length:', N_DAYS


    seq_df = seq.to_frame('seq')
    seq_df['weekday'] = None
    seq_df['locationSet'] = None
    for i in seq_df.index.values:
        seq_df.loc[i,'weekday'] = i.weekday()
        temp_set = set(seq_df.loc[i,'seq'])
        seq_df.loc[i,'locationSet'] = [x for x in temp_set if x!= -99]
    seq_summary_file = SUMMARY_PATH + fileID + '_daySequence_summary.csv'
    seq_df.to_csv(seq_summary_file)

    # Seperate data into training set and test set
    seq_train = seq1[0:N_DAYS]
    seq_test = seq1[N_DAYS:]
    end_date_train = seq1.index.values[N_DAYS-1]
    start_date_train = seq1.index.values[0]

    #############################################################
    # Location purpose
    home_ID = 0
    work_ID = 1
    ##############################################################
    # 6. Determine number of locations and parameters to calibrate
    # Add all the locations visited and frequency into dictionary
    location_dict = {}
    work_days_dict = {}
    for i in range(7):
        work_days_dict[i] = []
    for i in seq_train.index.values:
        weekday = i.weekday()
        day_seq = seq_train[i]
        seq_set = set(day_seq)
        if work_ID in seq_set:
            work_days_dict[weekday].append(1)
        else:
            work_days_dict[weekday].append(0)
        for loc in seq_set:
            if loc != -99:
                if loc in location_dict:
                    location_dict[loc] += 1
                else:
                    location_dict[loc] = 1

    # work_days_boolean = {}
    # for i in work_days_dict:
    #     work_days_boolean[i] = round( sum(work_days_dict[i]) * 1.0 / len(work_days_dict[i]))
    # print 'Work days:',work_days_boolean

    # Only keep the locations been visited certain times
    location_frequent_dict = {}
    NUMBER_VISIT_BUFFER = 2
    for loc in location_dict:
        if location_dict[loc] > NUMBER_VISIT_BUFFER:
            location_frequent_dict[loc] = location_dict[loc]
    print 'Frequent visited locations:',location_frequent_dict

    # Determine non-work activity pattern
    # Pattern based on day of week
    non_work_locations = list({x for x in location_frequent_dict if x != home_ID and x!= work_ID})
    nonwork_days_dict = {}
    for ID in non_work_locations:
        for day_of_week in range(7):
            nonwork_days_dict[(ID,day_of_week)] = []

    for i in seq_train.index.values:
        weekday = i.weekday()
        for ID in non_work_locations:
            if ID in seq_train[i]:
                nonwork_days_dict[(ID,weekday)].append(1)
            else:
                nonwork_days_dict[(ID, weekday)].append(0)

    nonwork_days_boolean = {}
    for ID in non_work_locations:
        for day_of_week in range(7):
            nonwork_days_boolean[(ID,day_of_week)] = round(sum(nonwork_days_dict[(ID,day_of_week)]) * 1.0 / len(nonwork_days_dict[(ID,day_of_week)]))
    # print 'Nonwork locations:',nonwork_days_boolean
    # print 'Nonwork locations visited on weekdays:',[x for x in nonwork_days_boolean if nonwork_days_boolean[x] > 0]

    # Pattern based on interval
    nonwork_interval_dict = {}
    nonwork_temp_index = {}
    for ID in non_work_locations:
        nonwork_interval_dict[ID] = []
        nonwork_temp_index[ID] = -1

    for i in range(len(seq_train)):
        seq_i = seq_train[i]
        for ID in non_work_locations:
            if ID in seq_i:
                if nonwork_temp_index[ID] == -1:
                    nonwork_temp_index[ID] = i
                else:
                    nonwork_interval_dict[ID].append(i-nonwork_temp_index[ID])
                    nonwork_temp_index[ID] = i
    # print 'Nonwork activity interval:', nonwork_interval_dict



    bestInterval = {}
    bestAccuracy = {}
    non_work_method = {}
    '''
    non_work_method represents how to predict if an activity is taken in a day
    1: weekday method. If taking the activity is based on the day of week. For location 2, if nonwork_days_boolean(2, weekday) is 1, take the activity, otherwise not taking
    2: frequency method. Taking the activity based on interval. The interval is stored in bestInterval[ID]
    '''


    for ID in non_work_locations:
        ID_common = DataProcessingFunctions_Android.most_common(nonwork_interval_dict[ID])
        start_index = 0
        for i in range(len(seq_train)):
            seq_i = seq_train[i]
            if ID in seq_i:
                start_index = i + 1
                break
        # print ID,'start_index:', seq_train.index.values[start_index]
        bestInterval[ID] = 0
        bestAccuracy[ID] = 0
        for interval in set(nonwork_interval_dict[ID]):
            if interval >= ID_common:
                currentAccuracy = intervalAccuracy(seq_train[start_index:], interval, ID)
                if currentAccuracy > bestAccuracy[ID]:
                    bestAccuracy[ID]  = currentAccuracy
                    bestInterval[ID] = interval

        day_of_week_accuracy = weekdayAccuracy(seq_train[start_index:],nonwork_days_boolean,ID )
        print ID, 'bestInterval:',bestInterval[ID],'bestAccuracy:',bestAccuracy[ID],'Day of week accuracy:',day_of_week_accuracy, 'Total days:',len(seq_train[start_index:])
        if (day_of_week_accuracy >= bestAccuracy[ID]-2):
            non_work_method[ID] = 1
        else:
            non_work_method[ID] = 2

    print 'Nonwork locations visited on weekdays:', [x for x in nonwork_days_boolean if nonwork_days_boolean[x] > 0]
    # print 'Nonwork method',non_work_method
    print 'Nonwork locations use method 2:',[x for x in non_work_method if non_work_method[x] == 2]
    ##############################################################
    # # 7. Getting information from training data
    # # Getting the observed travel time and preferred arrival time
    # # Read trip file
    # df_trip = DataProcessingFunctions_Android.readTripsFromCSV(trip_file)
    #
    # # Load all the observed travel time and departure times in the trip file (only the training days)
    # dict_T = {}
    # dict_arrivalT = {}
    # dict_endT = {}
    # dict_duration_raw = {}
    #
    # df_trip['date'] = None
    # for i in df_trip.index.values:
    #     df_trip.loc[i,'date'] = df_trip.loc[i,'end_time'].to_datetime().date()
    #
    # df_trip_train = df_trip[df_trip['date'] <= end_date_train]
    #
    # for i in df_trip.index.values:
    #     end_date = df_trip.loc[i,'date']
    #     if end_date > end_date_train:
    #         break
    #     O_ID = df_trip.loc[i,'locationID_O']
    #     D_ID = df_trip.loc[i,'locationID_D']
    #     OD_pair = (O_ID, D_ID)
    #     travel_time = df_trip.loc[i,'trip_time'].seconds/60.0
    #     duration = df_trip.loc[i,'duration_D'].seconds/60.0
    #     # Update travel time dictionary
    #     if OD_pair not in dict_T:
    #         dict_T[OD_pair] = [travel_time]
    #     else:
    #         dict_T[OD_pair].append(travel_time)
    #     if D_ID in location_frequent_dict:
    #         if D_ID not in dict_duration_raw:
    #             dict_duration_raw[D_ID] = [duration]
    #         else:
    #             dict_duration_raw[D_ID].append(duration)
    #     # Update preferred arrival time list
    #     arrival_time = df_trip.loc[i,'end_time']
    #     arrival_time_t = arrival_time.hour * 60 + arrival_time.minute + arrival_time.second/60.0
    #     if D_ID not in dict_arrivalT:
    #         dict_arrivalT[D_ID] = [arrival_time_t]
    #     else:
    #         dict_arrivalT[D_ID].append(arrival_time_t)
    #
    # dict_arrivalT[work_ID] = []
    # dict_endT[work_ID] = []
    # df_trip_work_start = df_trip_train[df_trip_train['locationID_D']== work_ID]
    # df_start_time = df_trip_work_start.groupby(['date']).apply(findWorkStartTime)
    # dict_arrivalT[work_ID] = list(df_start_time['workStartTime'])
    #
    # df_trip_work_end = df_trip_train[df_trip_train['locationID_O']== work_ID]
    # df_end_time = df_trip_work_end.groupby(['date']).apply(findWorkEndTime)
    # dict_endT[work_ID] = list(df_end_time['workEndTime'])
    #
    # df_home = df_trip_train[df_trip_train['locationID_O']== home_ID]
    # df_home_number = df_home.groupby(['date']).apply(findHomeWorkVistTimes)
    # max_home_visits = df_home_number['visitTimes'].max()+1
    #
    # df_work = df_trip_train[df_trip_train['locationID_O']== work_ID]
    # df_work_number = df_work.groupby(['date']).apply(findHomeWorkVistTimes)
    # max_work_visits = df_work_number['visitTimes'].max()
    #
    # # Avg day distance for frequent locations
    # dict_frequent_day_distance = {}
    # for ID in location_frequent_dict:
    #     dict_frequent_day_distance[ID] = N_DAYS * 1.0/location_frequent_dict[ID]
    # #
    # ##############################################################
    # # Model set up
    # home_locations = range(100, 100+max_home_visits)
    # work_locations = range(200,200+max_work_visits)
    # work_locations_other = [i for i in work_locations if i != 200]
    # non_home_locations = work_locations + non_work_locations
    # print 'non-work locations:',non_work_locations
    # print 'non_home locations:', non_home_locations
    # locations = home_locations + work_locations + non_work_locations
    # penalty_locations = [work_locations[0]] + non_work_locations
    #
    # T = 1440
    # M = 2000
    #
    # # Capture observed arrival departure time information
    # work_end_time = findMaxProKernel(dict_endT[work_ID],MIN_STARTTIME,MAX_STARTTIME)
    # preferred_arrival = {}
    # for i in dict_arrivalT:
    #     preferred_arrival[i] = findMaxProKernel(dict_arrivalT[i],MIN_STARTTIME,MAX_STARTTIME)
    # preferred_arrival = replaceHomeWorkLocations(preferred_arrival,home_ID,home_locations)
    # preferred_arrival = replaceHomeWorkLocations(preferred_arrival,work_ID,work_locations)
    #
    # travel_time_dict = {}
    # for OD in dict_T:
    #     if OD[0] in location_frequent_dict and OD[1] in location_frequent_dict:
    #         travel_time_dict[OD] = sum(dict_T[OD]) / len(dict_T[OD])
    #
    # travel_time_dict = replaceHomeWorkLinks(travel_time_dict,home_ID,home_locations)
    # travel_time_dict = replaceHomeWorkLinks(travel_time_dict,work_ID,work_locations)
    #
    # dict_duration = {}
    # for ID in dict_duration_raw:
    #     dict_duration[ID] = findMaxProKernel(dict_duration_raw[ID],MIN_DURATION,MAX_DURATION)
    # print 'Duration of non-work activities:',dict_duration
##############################################################
# 8. Generate the initial parameter vector
# Parameter explanation
# 1. home_u
# 2. work_u
# 3. work early penalty
# 4. work late penalty
# Afterwards are the parameters for non-work activities. For each activity, there are 4 parameters
#     1. Beta, which is the utility decrease rate by minute (marginal utility decreases along with time)
#     2. deltaU, which is the need growth rate
#     3. early penalty
#     4. late penalty
input_parameters = [1,100,1,1,1,11,1,1,1,3,1,1]
bounds = [[0,10],[0,100,],[0,5],[0,5],[0,10],[0,20],[0,5],[0,5],[0,10],[0,20],[0,5],[0,5]]

def similarityEvaluationForParameterSet(input_parameters):
    # Travel disutility rhoT (we can fix this)
    travel_u = 0.1
    # Home parameter: home utility
    home_u = input_parameters[0]
    # Work parameters: beta0 (work utility), rho1, rho2 (early late penalty)
    # beta1 and beta2 can be calculated based on observed data
    work_u = input_parameters[1]
    early_penalty = {}
    late_penalty = {}
    early_penalty[work_locations[0]] = input_parameters[2]
    late_penalty[work_locations[0]] = input_parameters[2]
    # Non-work parameters: beta (utility decrease rate), deltaU(daily utility increase rate), rho1, rho2 (early late penalty)
    # beta = {}
    deltaU = {}
    parameter_index = 3
    for ID in non_work_locations:
        # beta[ID] = input_parameters[parameter_index]
        # parameter_index += 1

        deltaU[ID] = input_parameters[parameter_index]
        parameter_index += 1
        early_penalty[ID] = input_parameters[parameter_index]
        # parameter_index += 1
        late_penalty[ID] = input_parameters[parameter_index]
        parameter_index += 1
        # beta[ID] = max((deltaU[ID] * 1.0 * dict_frequent_day_distance[ID] - home_u),0) * 1.0 / dict_duration[ID]
    print 'deltaU:',deltaU
    # print 'beta:',beta
    initialU = {}
    ##############################################################
    # Set other parameters based on given parameter
    for i in non_work_locations:
        initialU[i] = calInitialU(i,df_trip_train,start_date_train,deltaU)
    ##############################################################
    # 9. Given the parameter vector, predict the activity travel patterns
    # For the first day
    current_index = 0
    current_day = seq1.index.values[current_index]
    current_weekday = current_day.weekday()
    work_u_today = work_u * work_days_boolean[current_weekday]

    # Add travel time
    # Filter travel time dictionary. Only considers those links that have been taken before
    links = list(travel_time_dict.keys())
    links_home1 = [x for x in links if x[1] == home_locations[0]]

    machine_start_time = time.time()
    # Create optimization model
    m = Model("routing")

    # Create decision variables
    node = m.addVars(locations,name="node",vtype = GRB.BINARY)
    link = m.addVars(links,name="link",vtype = GRB.BINARY)
    start_time = m.addVars(locations, lb=MIN_STARTTIME, ub = MAX_STARTTIME,name="start_time")
    end_time = m.addVars(locations, lb=MIN_STARTTIME, ub = MAX_STARTTIME,name="end_time")

    # Create intermediate variables
    au_early = m.addVars(penalty_locations, lb=0,ub=MAX_DURATION,name="au_early")
    au_late = m.addVars(penalty_locations, lb=0,ub=MAX_DURATION,name="au_late")
    au_b = m.addVars(links,name="au_b",vtype = GRB.BINARY)
    au_a_list = [1,2]
    au_a = m.addVars(work_locations,au_a_list,name="au_a")

    # Set objective
    obj = QuadExpr()
    # obj += sum((initialU[i] * ( end_time[i] - start_time[i] ) - 0.5*beta[i] * (end_time[i] - start_time[i] )* ( end_time[i] - start_time[i] ))
    #     for i in non_work_locations)
    obj += sum((initialU[i] * (end_time[i] - start_time[i]) )
               for i in non_work_locations)
    obj += sum((end_time[i] - start_time[i] ) * home_u
        for i in home_locations)
    obj += sum((au_a[i,1] - au_a[i,2] ) * work_u_today
        for i in work_locations)
    obj += -sum((au_early[i] * early_penalty[i] + au_late[i] * late_penalty[i])
        for i in penalty_locations)
    obj += -sum((travel_time_dict[i] * link[i] * travel_u)
        for i in links)
    m.setObjective(obj, GRB.MAXIMIZE)

    # Add constraint: abs for work
    m.addConstrs(
        (au_a[i,1] <= work_end_time
        for i in work_locations),"work_abs_1")
    m.addConstrs(
        (au_a[i,1] <= end_time[i]
        for i in work_locations),"work_abs_2")
    m.addConstrs(
        (au_a[i,2] >= preferred_arrival[work_locations[0]]
        for i in work_locations),"work_abs_3")
    m.addConstrs(
        (au_a[i,2] >= start_time[i]
        for i in work_locations),"work_abs_4")

    if len(work_locations_other) > 0:
        m.addConstrs(
            (node[i] <= node[work_locations[0]]
             for i in work_locations_other), "work_abs_5")

    # Add constraint: start location constraint
    m.addConstr(
        (node[home_locations[0]] - 1 == 0)
        ,"start_location1")
    m.addConstr(
        (start_time[home_locations[0]] == MIN_STARTTIME)
        ,"start_location2")
    m.addConstrs(
        (link[i] == 0
        for i in links_home1),"start_location3")


    # Home location sequence
    if len(home_locations) > 2:
        m.addConstrs(
            (node[home_locations[i]] >= node[home_locations[i+1]]
             for i in range(len(home_locations)-2)), "home_location_sequence")

    # Add constraint: network flow constraint
    m.addConstrs(
        (link.sum('*',i) <=node[i]
        for i in locations),"inflow")
    m.addConstrs(
        (link.sum(i,'*') <=node[i]
        for i in locations),"outflow")

    # Add constraint: follow conservation, for activities other than home, inflow = outflow
    m.addConstrs(
        (link.sum(i, '*') == link.sum('*',i)
         for i in non_home_locations), "in out flow balance")

    # Add constraint: it's a route instead of a circle
    m.addConstr(
        (node.sum('*') - link.sum('*','*')==1)
        ,"route_constraint")

    # Add constraint: time consistency constraint
    m.addConstrs(
        ((link[i] - au_b[i] <= 0)
        for i in links),"time_consistency_1")
    m.addConstrs(
        ((end_time[i[0]] + travel_time_dict[i] - start_time[i[1]] + (1-au_b[i])* M >= 0)
        for i in links),"time_consistency_2")
    m.addConstrs(
        ((end_time[i[0]] + travel_time_dict[i] - start_time[i[1]] - (1-au_b[i])* M <= 0)
        for i in links),"time_consistency_3")
    m.addConstrs(
        ((start_time[i] <= end_time[i])
        for i in locations),"time_consistency_4")

    m.addConstrs(
        ((end_time[i]- start_time[i] - dict_duration[i] - (1-node[i]) * M  <= 0)
         for i in non_work_locations), "time_consistency_5")

    m.addConstrs(
        ((end_time[i] - start_time[i] - dict_duration[i] +(1- node[i]) * M >= 0)
         for i in non_work_locations), "time_consistency_6")
        #Add constraint: total time budget
    constr_total_time = LinExpr()
    constr_total_time += end_time.sum('*') - start_time.sum('*')
    constr_total_time += sum(link[i]*travel_time_dict[i] for i in links)
    m.addConstr(
        constr_total_time == T
        ,"total_time_budget")

    # Add constraint: fix the activity start time and end time for not active activities
    m.addConstrs(
        ((start_time[i] - preferred_arrival[i] + M*node[i] >= 0)
        for i in locations),"fix_not_active_activities1")
    m.addConstrs(
        ((start_time[i] - preferred_arrival[i] - M*node[i] <= 0)
        for i in locations),"fix_not_active_activities2")

    m.addConstrs(
        ((end_time[i] - preferred_arrival[i] + M*node[i] >= 0)
        for i in locations),"fix_not_active_activities3")
    m.addConstrs(
        ((end_time[i] - preferred_arrival[i] - M*node[i] <= 0)
        for i in locations),"fix_not_active_activities4")

    # Add constraint: start early or late
    m.addConstrs(
        ((start_time[i]+ au_early[i] - au_late[i] - preferred_arrival[i] == 0)
        for i in penalty_locations),"early_late"
    )

    # Compute optimal solution
    m.setParam('OutputFlag',False)
    m.optimize()
    # # Print solution
    # if m.status == GRB.Status.OPTIMAL:
    #     for v in m.getVars()[0:90]:
    #         print(v.varName, v.x)

    predict_day_seq = convertGurobiSolutionToDaySeq(node, start_time, end_time)
    observed_day_seq = daySeqRemoveTravel(seq_train[current_day])
    similarity_score = DataProcessingFunctions_Android.sequenceAlignmentBio(predict_day_seq,observed_day_seq)

    for current_index in range(1,N_DAYS):
        current_day = seq1.index.values[current_index]
        previous_day = seq1.index.values[current_index - 1]
        # print current_day
        current_weekday = current_day.weekday()
        work_u_today = work_u * work_days_boolean[current_weekday]
        for ID in non_work_locations:
            initialU[ID] = updateInitialUtility(ID, initialU,seq_train[previous_day], deltaU)
        print 'initialU:',initialU
        # print 'Work u today:', work_u_today,initialU
        # Update objective
        obj = QuadExpr()
        # obj += sum((initialU[i] * (end_time[i] - start_time[i]) - 0.5 * beta[i] * (end_time[i] - start_time[i]) * (
        # end_time[i] - start_time[i]))
        #            for i in non_work_locations)
        obj += sum((initialU[i] * (end_time[i] - start_time[i]))
                   for i in non_work_locations)
        obj += sum((end_time[i] - start_time[i]) * home_u
                   for i in home_locations)
        obj += sum((au_a[i, 1] - au_a[i, 2]) * work_u_today
                   for i in work_locations)
        obj += -sum((au_early[i] * early_penalty[i] + au_late[i] * late_penalty[i])
                    for i in penalty_locations)
        obj += -sum((travel_time_dict[i] * link[i] * travel_u)
                    for i in links)
        m.setObjective(obj, GRB.MAXIMIZE)
        m.optimize()
        predict_day_seq = convertGurobiSolutionToDaySeq(node, start_time, end_time)
        print predict_day_seq
        observed_day_seq = daySeqRemoveTravel(seq_train[current_day])
        # print 'predicted activity set:',set(predict_day_seq),'observed activity set:',set(observed_day_seq)
        similarity_score_today = DataProcessingFunctions_Android.sequenceAlignmentBio(predict_day_seq, observed_day_seq)
        similarity_score += similarity_score_today
        # print 'similarity score today:',similarity_score_today, 'total:',similarity_score
    print 'Total:',similarity_score,'input parameters',input_parameters
    return -similarity_score

machine_start_time = time.time()

Flag_ID = 1
for trip_file in tripList:
    print Flag_ID,'/',len(tripList)
    Flag_ID += 1
    evaluatePersonFile(trip_file)

# trip_file = 'App_data/Processed_Data/GPS_trips_processed/09a963d1-d7cb-42ed-9825-822e1dec2d57_daySequence.csv'
# evaluatePersonFile(trip_file)
##############################################################
# SPSA Optimization
input_parameters = [6,10,1,1.2,1,0,0]

input_parameters = [6,10,1,1.2,1,0.4957,1]
# # bounds = [[0,10],[0,100,],[0,5],[0,5],[0,10],[0,20],[0,5],[0,5],[0,10],[0,20],[0,5],[0,5]]
# similarityEvaluationForParameterSet(input_parameters)

# input_parameters = []
# bounds = []
# # Add home parameters
# input_parameters.append(7)
# bounds.append([0.1,10])
# # Add work parameters
# input_parameters = input_parameters + [10,1]
# bounds = bounds + [[0.1,20],[0.1,20]]
# # Add nonwork parameters
# for i in non_work_locations:
#     input_parameters = input_parameters + [0,1]
#     bounds = bounds + [[0,20],[0.1,20]]
# similarityEvaluationForParameterSet(input_parameters)

# res = minimizeSPSA_Liang(similarityEvaluationForParameterSet, bounds=bounds, x0=input_parameters, niter=30, paired=False)
# res = minimizeSPSA_Customize(similarityEvaluationForParameterSet, bounds=bounds, x0=input_parameters, niter=1000, paired=False)
# print res
##############################################################
# Brute optimization
# rranges = (slice(0,2,1),slice(0,50,10),slice(0, 2, 1), slice(0, 2, 1),slice(0,10,5),slice(0,20,5),slice(0,2,1),slice(0,2,1),slice(0,10,5),slice(0,10,2),slice(0,2,1),slice(0,2,1))
# res = optimize.brute(similarityEvaluationForParameterSet,rranges,full_output=True, finish=None)
##############################################################
machine_elapsed_time = time.time() - machine_start_time
print machine_elapsed_time

