import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import *
import os
import shutil
import random

parser = argparse.ArgumentParser()

parser.add_argument('--l3adjdatacluster',type=str,default='data/PEMS08/adj_mat_cluster2.pkl',help='adj data path')

parser.add_argument('--l3transmit',type=str,default='data/PEMS08/pems08_transmit2.csv',help='data path')

parser.add_argument('--l3cluster_nodes',type=int,default=30,help='number of clusterl3')

parser.add_argument('--device',type=str,default='cuda:0',help='')

parser.add_argument('--data',type=str,default='data/PEMS08',help='data path')

parser.add_argument('--adjdata',type=str,default='data/PEMS08/adj_mat.pkl',help='adj data path')

parser.add_argument('--adjdatacluster',type=str,default='data/PEMS08/adj_mat_cluster.pkl',help='adj data path')

parser.add_argument('--transmit',type=str,default='data/PEMS08/pems08_transmit.csv',help='data path')

parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--in_dim_cluster',type=int,default=2,help='inputs dimension')

parser.add_argument('--num_nodes',type=int,default=170,help='number of nodes')

parser.add_argument('--cluster_nodes',type=int,default=90,help='number of cluster')

parser.add_argument('--batch_size',type=int,default=16,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')

parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')

parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument("--force", type=str, default=False,help="remove params dir", required=False)

parser.add_argument('--save',type=str,default='./garage/PEMS08/baselines',help='save path')

parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--model',type=str,default='H_GCN_wh',help='adj type')
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')

args = parser.parse_args()
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def main():
    # Load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    sensor_ids_cluster, sensor_id_to_ind_cluster, adj_mx_cluster = util.load_adj(args.adjdatacluster, args.adjtype)
    l3_sensor_ids_cluster, l3_sensor_id_to_ind_cluster, l3_adj_mx_cluster = util.load_adj(args.l3adjdatacluster, args.adjtype)

    dataloader = util.load_dataset_cluster_l3(args.data, args.batch_size, args.batch_size, args.batch_size)
    print("dataloader finished")
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    supports_cluster = [torch.tensor(i).to(device) for i in adj_mx_cluster]
    l3_supports_cluster = [torch.tensor(i).to(device) for i in l3_adj_mx_cluster]

    transmit_np = np.float32(np.loadtxt(args.transmit, delimiter=','))
    transmit = torch.tensor(transmit_np).to(device)
    l3_transmit_np = np.float32(np.loadtxt(args.l3transmit, delimiter=','))
    l3_transmit = torch.tensor(l3_transmit_np).to(device)

    if args.model=='MultiGran-STGCN':
        engine = trainer13(args.in_dim, args.in_dim_cluster, args.seq_length, args.num_nodes, args.cluster_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, supports_cluster, transmit, args.decay,
                           args.l3cluster_nodes, l3_supports_cluster, l3_transmit)

    # Load the best model weights
    engine.model.load_state_dict(torch.load('weight_final.pth'))
    engine.model.eval()
    
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]
    total_inference_time = 0
    total_samples = 0

    for iter, (recent_x, day_x, week_x, y, recent_x_cluster, day_x_cluster, week_x_cluster, y_cluster, recent_x_cl3, day_x_cl3, week_x_cl3, y_cl3) in enumerate(dataloader['test_loader_cluster_l3'].get_iterator()):
        recent_x = torch.Tensor(recent_x).to(device)
        day_x = torch.Tensor(day_x).to(device)
        week_x = torch.Tensor(week_x).to(device)
        recent_x_cluster = torch.Tensor(recent_x_cluster).to(device)
        day_x_cluster = torch.Tensor(day_x_cluster).to(device)
        week_x_cluster = torch.Tensor(week_x_cluster).to(device)
        recent_x_cl3 = torch.Tensor(recent_x_cl3).to(device)
        day_x_cl3 = torch.Tensor(day_x_cl3).to(device)
        week_x_cl3 = torch.Tensor(week_x_cl3).to(device)

        # Transpose the tensors to match the model's expected input shape
        recent_x = recent_x.transpose(1, 3)
        day_x = day_x.transpose(1, 3)
        week_x = week_x.transpose(1, 3)
        recent_x_cluster = recent_x_cluster.transpose(1, 3)
        day_x_cluster = day_x_cluster.transpose(1, 3)
        week_x_cluster = week_x_cluster.transpose(1, 3)
        recent_x_cl3 = recent_x_cl3.transpose(1, 3)
        day_x_cl3 = day_x_cl3.transpose(1, 3)
        week_x_cl3 = week_x_cl3.transpose(1, 3)

        batch_size = recent_x.shape[0]
        total_samples += batch_size

        s1 = time.time()
        with torch.no_grad():
            preds, _, _, _ = engine.model(recent_x, day_x, week_x, recent_x_cluster, day_x_cluster, week_x_cluster, recent_x_cl3, day_x_cl3, week_x_cl3)
            preds = preds.transpose(1, 3)
        s2 = time.time()
        inference_time = s2 - s1

        outputs.append(preds.squeeze())
        total_inference_time += inference_time
    
    print(f"CPU total Inference time: {total_inference_time:.4f} seconds")
    average_inference_time_per_sample = total_inference_time / total_samples
    print(f"Average CPU Inference Time per Sample: {average_inference_time_per_sample:.6f} seconds")

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae, amape, armse = [], [], []
    prediction = yhat
    for i in range(12):
        pred = prediction[:, :, i]
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
    
    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))

if __name__ == "__main__":
    main()
