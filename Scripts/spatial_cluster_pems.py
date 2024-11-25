import numpy as np
import pandas as pd
import re
import math
import os

def set_working_directory():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    print("Current working directory set to:", dname)

set_working_directory()

def find_min_cluster_distance(distance, transmit, cluster_nodes):
    min_cluster_distances = np.full((cluster_nodes, cluster_nodes), np.inf)
    
    for i in range(cluster_nodes):
        for j in range(cluster_nodes):
            if i != j:
                cluster_i_nodes = np.where(transmit[:, i] == 1)[0]
                cluster_j_nodes = np.where(transmit[:, j] == 1)[0]
                for node_i in cluster_i_nodes:
                    for node_j in cluster_j_nodes:
                        if distance[node_i, node_j] < min_cluster_distances[i, j]:
                            min_cluster_distances[i, j] = distance[node_i, node_j]
    return min_cluster_distances


road_mat = np.zeros((170, 4))
shape = road_mat.shape

cluster_nodes = 30

distance = np.full((shape[0], shape[0]), 99999)
np.fill_diagonal(distance, 0)

distance_info = pd.read_csv('pems08_distance.csv')

for index, row in distance_info.iterrows():
    from_node = int(row['from'])
    to_node = int(row['to'])
    cost = row['cost']
    distance[from_node, to_node] = cost
    distance[to_node, from_node] = cost
    #print(cost)

print(distance)

adj1=np.zeros((shape[0],shape[0]))
count=0

adj2=np.zeros((shape[0],shape[0]))


for index, row in distance_info.iterrows():
    from_node = int(row['from'])
    print(from_node)
    to_node = int(row['to'])
    var = np.var(distance[from_node, :])
    adj1[from_node, to_node] = np.exp(-distance[from_node, to_node] / var)
    adj1[to_node, from_node] = np.exp(-distance[to_node, from_node] / var)
    print(adj1[from_node, to_node])

# for i in range(shape[0]):
#     order=np.argsort(distance[i,:])
#     var=np.var(distance[i,:])
#     for j in range(5):
#         if np.exp(-distance[i,order[j]]/var)>0.95:
#             adj1[i,order[j]]=np.exp(-distance[i,order[j]]/var)
#             print(adj1[i,order[j]])
#             count+=1
            
for i in range(shape[0]):
    var=np.var(distance[i,:])
    for j in range(shape[0]):            
        adj2[i,j]=np.exp(-distance[i,j]/var)


def calLaplacianMatrix(adjacentMatrix):

    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
Laplacian=calLaplacianMatrix(adj2)
lam, H = np.linalg.eig(Laplacian) # H'shape is n*n
lam=lam.real
H=H.real
H=H[:,0:20]


from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, AffinityPropagation

def spKmeans(H, cluster_nodes):
    sp_kmeans = KMeans(n_clusters=cluster_nodes, n_init=30, random_state=9).fit(H)
    return sp_kmeans.labels_

def Aggs(data, cluster_nodes):
    data = np.squeeze(data)
    dbscan = AgglomerativeClustering(n_clusters=cluster_nodes).fit(data)
    return dbscan.labels_

def Spect(data, cluster_nodes):
    data = np.squeeze(data)
    Sp = SpectralClustering(n_clusters=cluster_nodes).fit(data)
    return Sp.labels_

def Affinity(data, cluster_nodes):
    data = np.squeeze(data)
    Af = AffinityPropagation().fit(data)
    return Af.labels_

data = np.load('pems08_flow.npz')['data']
data =np.transpose(data,(1,0,2))

idx = spKmeans(H, cluster_nodes)

print(idx.shape)

road_cluster_id = np.zeros(cluster_nodes)
for i in range(cluster_nodes):
    road_cluster_id[i] = i
np.savetxt('pems08_graph_sensor_cluster_ids2.txt', road_cluster_id, delimiter=',')


transmit = np.zeros((shape[0], cluster_nodes))
for i in range(shape[0]):
    transmit[i, idx[i]] = 1

adj_pool = np.dot(np.dot(transmit.transpose(), adj1), transmit)
dist_pool = []

for i in range(cluster_nodes):
    for j in range(cluster_nodes):         
       if adj_pool[i, j] != 0:
            dist_pool.append([i, j, 1])

distance2 = np.mat(dist_pool)   
distance2 = pd.DataFrame(distance2)
distance2.columns = ['from', 'to', 'distance']
distance2.to_csv('pems08_distance_cluster2.csv', index=False) 

np.savetxt('pems08_transmit2.csv', transmit, delimiter=',')

shape=data.shape
print("shape",shape)
print("shape[0]",shape[0])
data_cluster=[]
for i in range(cluster_nodes):
    listdata=[]
    for j in range(shape[0]):
        if idx[j]==i:
            listdata.append(data[j,:,:])
    data_pool1=np.stack(listdata,0)       
    data_pool2=np.min(data_pool1,0)
    data_pool3=np.mean(data_pool1,0)
    data_pool4=np.max(data_pool1,0)
    data_pool5=np.concatenate((data_pool2,data_pool3),-1)
    data_cluster.append(data_pool5)
data_cluster=np.stack(data_cluster,0)
np.savez_compressed('pems08_original_data_cluster2', data=data_cluster)
