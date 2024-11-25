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
seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main():
    #load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    sensor_ids_cluster, sensor_id_to_ind_cluster, adj_mx_cluster = util.load_adj(args.adjdatacluster,args.adjtype)

    l3_sensor_ids_cluster, l3_sensor_id_to_ind_cluster, l3_adj_mx_cluster = util.load_adj(args.l3adjdatacluster,args.adjtype)

    dataloader = util.load_dataset_cluster_l3(args.data, args.batch_size, args.batch_size, args.batch_size)
    print("dataloader finished")

    # print("Sensor IDs:", sensor_ids)
    # print("Sensor ID to Index mapping:", sensor_id_to_ind)
    # print("Adjacency Matrix Shape:", adj_mx[0].shape if isinstance(adj_mx, list) else adj_mx.shape)
    # print("Cluster Sensor IDs:", sensor_ids_cluster)
    # print("Cluster Sensor ID to Index mapping:", sensor_id_to_ind_cluster)
    # print("Cluster Adjacency Matrix Shape:", adj_mx_cluster[0].shape if isinstance(adj_mx_cluster, list) else adj_mx_cluster.shape)
    # print("L3 Cluster Sensor IDs:", l3_sensor_ids_cluster)
    # print("L3 Cluster Sensor ID to Index mapping:", l3_sensor_id_to_ind_cluster)
    # print("L3 Cluster Adjacency Matrix Shape:", l3_adj_mx_cluster[0].shape if isinstance(l3_adj_mx_cluster, list) else l3_adj_mx_cluster.shape)

    #scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    supports_cluster = [torch.tensor(i).to(device) for i in adj_mx_cluster]

    l3_supports_cluster = [torch.tensor(i).to(device) for i in l3_adj_mx_cluster]

    transmit_np=np.float32(np.loadtxt(args.transmit,delimiter=','))
    transmit=torch.tensor(transmit_np).to(device)

    l3_transmit_np=np.float32(np.loadtxt(args.l3transmit,delimiter=','))
    l3_transmit=torch.tensor(l3_transmit_np).to(device)
    
    print("args:")
    print(args)
    
    if args.model=='MultiGran-STGCN':
        engine = trainer13( args.in_dim,args.in_dim_cluster, args.seq_length, args.num_nodes,args.cluster_nodes,args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports,supports_cluster,transmit,args.decay,
                         args.l3cluster_nodes, l3_supports_cluster, l3_transmit
                         )

    params_path=args.save+"/"+args.model
    if os.path.exists(params_path) and not args.force:
        raise SystemExit("Params folder exists! Select a new params path please!")
    else:
        if os.path.exists(params_path):
            shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Create params directory %s' % (params_path))

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        
        dataloader['train_loader_cluster_l3'].shuffle()

        for iter, (recent_x, day_x, week_x, y, recent_x_cluster, day_x_cluster, week_x_cluster, y_cluster, recent_x_cl3, day_x_cl3, week_x_cl3, y_cl3) in enumerate(dataloader['train_loader_cluster_l3'].get_iterator()):
            recent_x = torch.Tensor(recent_x).to(device)
            day_x = torch.Tensor(day_x).to(device)
            week_x = torch.Tensor(week_x).to(device)
            y = torch.Tensor(y).to(device)
            
            recent_x_cluster = torch.Tensor(recent_x_cluster).to(device)
            day_x_cluster = torch.Tensor(day_x_cluster).to(device)
            week_x_cluster = torch.Tensor(week_x_cluster).to(device)
            y_cluster = torch.Tensor(y_cluster).to(device)
            
            recent_x_cl3 = torch.Tensor(recent_x_cl3).to(device)
            day_x_cl3 = torch.Tensor(day_x_cl3).to(device)
            week_x_cl3 = torch.Tensor(week_x_cl3).to(device)
            y_cl3 = torch.Tensor(y_cl3).to(device)
            
            recent_x = recent_x.transpose(1, 3)
            day_x = day_x.transpose(1, 3)
            week_x = week_x.transpose(1, 3)
            train_y = y.transpose(1, 3)
            
            recent_x_cluster = recent_x_cluster.transpose(1, 3)
            day_x_cluster = day_x_cluster.transpose(1, 3)
            week_x_cluster = week_x_cluster.transpose(1, 3)
            
            recent_x_cl3 = recent_x_cl3.transpose(1, 3)
            day_x_cl3 = day_x_cl3.transpose(1, 3)
            week_x_cl3 = week_x_cl3.transpose(1, 3)

            metrics = engine.train(recent_x, day_x, week_x, train_y[:, 0, :, :], recent_x_cluster, day_x_cluster, week_x_cluster, recent_x_cl3, day_x_cl3, week_x_cl3)
            
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
        
        t2 = time.time()
        train_time.append(t2 - t1)

        #validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        
        # validation
        for iter, (recent_x, day_x, week_x, y, recent_x_cluster, day_x_cluster, week_x_cluster, y_cluster, recent_x_cl3, day_x_cl3, week_x_cl3, y_cl3) in enumerate(dataloader['val_loader_cluster_l3'].get_iterator()):
            recent_x = torch.Tensor(recent_x).to(device)
            day_x = torch.Tensor(day_x).to(device)
            week_x = torch.Tensor(week_x).to(device)
            recent_x_cluster = torch.Tensor(recent_x_cluster).to(device)
            day_x_cluster = torch.Tensor(day_x_cluster).to(device)
            week_x_cluster = torch.Tensor(week_x_cluster).to(device)
            recent_x_cl3 = torch.Tensor(recent_x_cl3).to(device)
            day_x_cl3 = torch.Tensor(day_x_cl3).to(device)
            week_x_cl3 = torch.Tensor(week_x_cl3).to(device)
            y = torch.Tensor(y).to(device)

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
            valid_y = y.transpose(1, 3)

            # Evaluate the model
            metrics = engine.eval(recent_x, day_x, week_x, valid_y[:, 0, :, :], recent_x_cluster, day_x_cluster, week_x_cluster, recent_x_cl3, day_x_cl3, week_x_cl3)
            
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])

        
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), params_path+"/"+args.model+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(params_path+"/"+args.model+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
    engine.model.eval()
    
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    
    realy = realy.transpose(1,3)[:,0,:,:]
    #print(realy.shape)

    # l3
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
        y = torch.Tensor(y).to(device)

        recent_x = recent_x.transpose(1, 3)
        day_x = day_x.transpose(1, 3)
        week_x = week_x.transpose(1, 3)
        recent_x_cluster = recent_x_cluster.transpose(1, 3)
        day_x_cluster = day_x_cluster.transpose(1, 3)
        week_x_cluster = week_x_cluster.transpose(1, 3)
        recent_x_cl3 = recent_x_cl3.transpose(1, 3)
        day_x_cl3 = day_x_cl3.transpose(1, 3)
        week_x_cl3 = week_x_cl3.transpose(1, 3)
        with torch.no_grad():
            preds, _, _, _ = engine.model(recent_x, day_x, week_x, recent_x_cluster, day_x_cluster, week_x_cluster, recent_x_cl3, day_x_cl3, week_x_cl3)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

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
        y = torch.Tensor(y).to(device)

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
        with torch.no_grad():
            _, spatial_at, spatial_at_l3, parameter_adj = engine.model(recent_x, day_x, week_x, recent_x_cluster, day_x_cluster, week_x_cluster, recent_x_cl3, day_x_cl3, week_x_cl3)
        break
            
        
    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    #print(yhat.shape)
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    amae = []
    amape = []
    armse = []
    prediction=yhat
    for i in range(12):
        pred = prediction[:,:,i]
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
    
    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(),params_path+"/"+args.model+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    prediction_path=params_path+"/"+args.model+"_prediction_results"
    ground_truth=realy.cpu().detach().numpy()
    prediction=prediction.cpu().detach().numpy()
    spatial_at=spatial_at.cpu().detach().numpy()
    spatial_at_l3=spatial_at_l3.cpu().detach().numpy()
    parameter_adj=parameter_adj.cpu().detach().numpy()
    np.savez_compressed(
            os.path.normpath(prediction_path),
            prediction=prediction,
            spatial_at=spatial_at,
            spatial_at_l3=spatial_at_l3,
            parameter_adj=parameter_adj,
            ground_truth=ground_truth
        )


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
