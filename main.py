# Author: Prajwal, Dhatwalia

from gurobipy import *
from math import *
from random import *
from sklearn.cluster import *
import matplotlib.pyplot as plt

def distance(coordinate1, coordinate2):
    distance = (coordinate1[0]-coordinate2[0])**2 + (coordinate1[1]-coordinate2[1])**2
    return sqrt(distance)

def random_coordinates():
    x = gauss(0.50,0.07)
    y = gauss(0.50,0.07)
    return (x,y)

no_of_patients = 1000
no_of_clusters = 40
no_of_sites = 20
no_of_hospitals = 10
# NOTE: We must select atmost no_of_hospitals from no_of_sites 

patient_locations = []
for i in range(no_of_patients):
    patient_locations.append(random_coordinates())

hospital_locations = []
for i in range(no_of_sites):
    hospital_locations.append(random_coordinates())

# Start clustering
kmeans = KMeans(n_clusters=no_of_clusters)
kmeans.fit(patient_locations)
cluster_centers = kmeans.cluster_centers_

hospital_assigned = {}
for hospital in range(no_of_sites):
    for cluster in range(no_of_clusters):
        if  distance(hospital_locations[hospital], cluster_centers[cluster]) < 0.9:
            hospital_assigned[hospital,cluster] = distance(hospital_locations[hospital], cluster_centers[cluster])

model = Model("Hospital Distance Minimizer")

# Variables
hospitals_chosen = []
for i in range(no_of_sites):
    hospitals_chosen.append(model.addVar(vtype=GRB.BINARY, name='hospitals_chosen['+str(i)+']'))

patients_assigned = {}
for hospital, cluster in hospital_assigned.keys():
    patients_assigned[hospital, cluster] = model.addVar(vtype=GRB.BINARY, name='patients_assigned['+str(hospital)+','+str(cluster)+']')

# Constraints
# Chosen hospitals are used
for hospital, cluster in hospital_assigned.keys():
    model.addConstr(patients_assigned[hospital, cluster] <= hospitals_chosen[hospital])

# Total hospitals must not exceed the no. of hospitals
total_hospitals = 0
for i in range(len(hospitals_chosen)):
    total_hospitals += hospitals_chosen[i]
model.addConstr(total_hospitals <= no_of_hospitals)

# One hospital per cluster
for i in range(no_of_clusters):
    clusters_assigned = 0
    for hospital,cluster in hospital_assigned.keys():
        if i == cluster:
            clusters_assigned += patients_assigned[hospital, cluster]
    model.addConstr(clusters_assigned == 1)


# Objective Function
total_distance = 0
for hospital, cluster in hospital_assigned.keys():
    total_distance += hospital_assigned[hospital, cluster] * patients_assigned[hospital, cluster]
model.setObjective(total_distance, GRB.MINIMIZE)

# Optimize
model.optimize()

# Plot and display the result
plt.scatter(*zip(*patient_locations), color='SkyBlue')
plt.scatter(*zip(*cluster_centers), color='Blue')
plt.scatter(*zip(*hospital_locations), color='Pink')

for hospital in hospital_assigned:
    if patients_assigned[hospital].x > 0.5:
        plt.scatter(*zip(hospital_locations[hospital[0]]), color='Red')
        plt.plot(*zip(hospital_locations[hospital[0]], cluster_centers[hospital[1]]), color='Black')

plt.show()
