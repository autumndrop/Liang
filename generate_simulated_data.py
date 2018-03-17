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

os.chdir(parent_path)
GENERATE_PATH = 'App_data/Generated_data/'
GENERATE_SEQ_PATH = 'App_data/Generated_data/daySequence/'
GENERATE_TRIP_PATH = 'App_data/Generated_data/trips/'
GENERATE_LOCATION_SEQUENCE = 'App_data/Generated_data/locationSequence/'
GENERATE_LOCATION_PLOT = 'App_data/Generated_data/locationSequence_plots/'

if not os.path.isdir(GENERATE_PATH):
    os.mkdir(GENERATE_PATH)
if not os.path.isdir(GENERATE_SEQ_PATH):
    os.mkdir(GENERATE_SEQ_PATH)
if not os.path.isdir(GENERATE_TRIP_PATH):
    os.mkdir(GENERATE_TRIP_PATH)
if not os.path.isdir(GENERATE_LOCATION_SEQUENCE):
    os.mkdir(GENERATE_LOCATION_SEQUENCE)
if not os.path.isdir(GENERATE_LOCATION_PLOT):
    os.mkdir(GENERATE_LOCATION_PLOT)


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

def convertNode(ID,home_locations,home_ID,work_locations,work_ID):
    node_ID = ID
    if node_ID in home_locations:
        node_ID = home_ID
    elif node_ID in work_locations:
        node_ID = work_ID
    return node_ID


fileID = 1
trip_file = GENERATE_TRIP_PATH + str(fileID) + '_trips.csv'
seq_file = GENERATE_SEQ_PATH + str(fileID) + '_daySequence.csv'
locSeq_file = GENERATE_LOCATION_SEQUENCE + str(fileID) + '_loactionSequence.csv'
# Location purpose
home_ID = 0
work_ID = 1
non_work_locations = [2,3]
locations = [home_ID] + [work_ID] + non_work_locations
work_days_boolean = {0:1,1:1,2:1,3:1,4:1,5:0,6:0}
travel_time_dict = {}
for loc1 in locations:
    for loc2 in locations:
        if loc1 != loc2:
            travel_time_dict[(loc1,loc2)] = 40

travel_time_dict[(0,2)] = 20
travel_time_dict[(2,0)] = 20
travel_time_dict[(1,3)] = 20
travel_time_dict[(3,1)] = 20

dict_duration = {}
dict_duration[2] = 30
dict_duration[3] = 40

preferred_arrival = {}
preferred_arrival[0] = 120
preferred_arrival[1] = 540
preferred_arrival[2] = 1140
preferred_arrival[3] = 1200
work_end_time = 1020

home_u = 8
work_u = 15
travel_u = 1
deltaU = {}
deltaU[2] = 3
deltaU[3] = 1

initialU = {}
initialU[2] = 3
initialU[3] = 1

early_penalty = {}
late_penalty = {}
early_penalty[1] = 5
late_penalty[1] = 5
early_penalty[2] = 1
late_penalty[2] = 1
early_penalty[3] = 1
late_penalty[3] = 1

##############################################################
# Model set up
max_home_visits = len(non_work_locations) + 2
max_work_visits = 1
home_locations = range(100, 100 + max_home_visits)
work_locations = range(200,200+max_work_visits)
work_locations_other = [i for i in work_locations if i != 200]
non_home_locations = work_locations + non_work_locations
print 'non-work locations:',non_work_locations
print 'non_home locations:', non_home_locations
locations = home_locations + work_locations + non_work_locations
penalty_locations = [work_locations[0]] + non_work_locations

T = 1440
M = 2000

start_date_train = datetime(2017,6,1)
N_DAYS = 60

dict_data = {}


current_day = start_date_train
current_weekday = current_day.weekday()
work_u_today = work_u * work_days_boolean[current_weekday]

travel_time_dict = replaceHomeWorkLinks(travel_time_dict,home_ID,home_locations)
travel_time_dict = replaceHomeWorkLinks(travel_time_dict,work_ID,work_locations)
preferred_arrival = replaceHomeWorkLocations(preferred_arrival,home_ID,home_locations)
preferred_arrival = replaceHomeWorkLocations(preferred_arrival,work_ID,work_locations)
early_penalty[work_locations[0]] = early_penalty[work_ID]
late_penalty[work_locations[0]] = late_penalty[work_ID]

links = list(travel_time_dict.keys())
links_home1 = [x for x in links if x[1] == home_locations[0]]

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
dict_data[current_day] = predict_day_seq


def convertGurobiSolutionToDF(df_data,current_df_index,previous_D_startTime, node,start_time, end_time):
    node_visited = [x for x in locations if node[x].X > 0]
    start_time_visited = [start_time[x].X for x in node_visited]
    end_time_visited = [end_time[x].X for x in node_visited]
    sequence = sorted(range(len(start_time_visited)), key=lambda k: start_time_visited[k])
    for i in range(len(sequence)-1):
        index1 = sequence[i]
        index2 = sequence[i+1]
        node1 = node_visited[index1]
        node2 = node_visited[index2]
        time1 = round(end_time_visited[index1])
        time1_date = datetime(current_day.year,current_day.month,current_day.day,int(time1/60),int(time1%60))
        time2 = round(start_time_visited[index2])
        time2_date = datetime(current_day.year, current_day.month, current_day.day, int(time2 / 60), int(time2 % 60))
        if previous_D_startTime == -1:
            previous_D_startTime = time2_date
        else:
            print time1_date
            print previous_D_startTime
            df_data.loc[current_df_index-1, 'duration_D'] = time1_date - previous_D_startTime
            previous_D_startTime = time2_date
        triptime = time2_date - time1_date
        df_data.loc[current_df_index, 'weekday'] = time1_date.weekday()
        df_data.loc[current_df_index,'start_time'] = time1_date
        df_data.loc[current_df_index,'end_time'] = time2_date
        df_data.loc[current_df_index,'trip_time'] = triptime
        df_data.loc[current_df_index,'locationID_D'] = convertNode(node2,home_locations,home_ID,work_locations,work_ID)
        df_data.loc[current_df_index,'locationID_O'] = convertNode(node1,home_locations,home_ID,work_locations,work_ID)
        current_df_index += 1
    return [df_data,current_df_index,previous_D_startTime]

df_data = pd.DataFrame(columns=['weekday','start_time','end_time','trip_time','duration_D','locationID_O','locationID_D'])
current_df_index = 0
previous_D_startTime = -1
[df_data,current_df_index,previous_D_startTime] = convertGurobiSolutionToDF(df_data,current_df_index,previous_D_startTime, node,start_time, end_time)


for current_index in range(1,N_DAYS):
    previous_day = current_day
    current_day = current_day + timedelta(days=1)
    current_weekday = current_day.weekday()
    work_u_today = work_u * work_days_boolean[current_weekday]
    for ID in non_work_locations:
        initialU[ID] = updateInitialUtility(ID, initialU,dict_data[previous_day], deltaU)
    print 'initialU:',initialU
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
    dict_data[current_day] = predict_day_seq
    [df_data,current_df_index,previous_D_startTime] = convertGurobiSolutionToDF(df_data,current_df_index,previous_D_startTime, node,start_time, end_time)
print dict_data

df_data.to_csv(trip_file)
df_seq = pd.DataFrame(dict_data.items(),columns = ['date','seq'])
df_seq = df_seq.sort_values('date')
df_seq = df_seq.reset_index(drop=True)
df_seq.to_csv(seq_file,header=False)
locationTimeInterval = 5
DataProcessingFunctions_Android.locationSequenceCreate(trip_file, locSeq_file, locationTimeInterval, START_HOUR)
DataProcessingFunctions_Android.plotASingleFile(locSeq_file,GENERATE_LOCATION_PLOT)
