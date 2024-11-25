import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter,LayerNorm,InstanceNorm2d
from utils import ST_BLOCK_0 
from utils import ST_BLOCK_4 
from utils import ST_BLOCK_5 
from utils import multi_gcn
from utils import GCNPool
from utils import Transmit
from utils import gate
from utils import gatel3


class ASTGCN(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(ASTGCN,self).__init__()
        self.block1=ST_BLOCK_0(in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_0(dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        self.final_conv=Conv2d(length,12,kernel_size=(1, dilation_channels),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)

        # Add weights for the three components
        self.W_recent = nn.Parameter(torch.randn(1, out_dim, 170, 1).to(device), requires_grad=True)
        self.W_day = nn.Parameter(torch.randn(1, out_dim, 170, 1).to(device), requires_grad=True)
        self.W_week = nn.Parameter(torch.randn(1, out_dim, 170, 1).to(device), requires_grad=True)
        
    def forward(self,input_recent, input_day, input_week):
        x=self.bn(input_recent)
        x_day=self.bn(input_day)
        x_week=self.bn(input_week)
        
        adj=self.supports[0]
        x,_,_ = self.block1(x,adj)
        x,d_adj,t_adj = self.block2(x,adj)
        x = x.permute(0,3,2,1)
        x = self.final_conv(x)#b,12,n,1

        x_day,_,_ = self.block1(x_day,adj)
        x_day,d_adj,t_adj = self.block2(x_day,adj)
        x_day = x_day.permute(0,3,2,1)
        x_day = self.final_conv(x_day)#b,12,n,1

        x_week,_,_ = self.block1(x_week,adj)
        x_week,d_adj,t_adj = self.block2(x_week,adj)
        x_week = x_week.permute(0,3,2,1)
        x_week = self.final_conv(x_week)#b,12,n,1

        x_mix = x * self.W_recent + x_day * self.W_day + x_week * self.W_week

        return x_mix,d_adj,t_adj
    


class LSTM(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(LSTM,self).__init__()
        self.lstm=nn.LSTM(in_dim,dilation_channels,batch_first=True)#b*n,l,c
        self.c_out=dilation_channels
        tem_size=length
        self.tem_size=tem_size
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        
        
    def forward(self,input):
        x=input
        shape = x.shape
        h = Variable(torch.zeros((1,shape[0]*shape[2],self.c_out))).cuda()
        c = Variable(torch.zeros((1,shape[0]*shape[2],self.c_out))).cuda()
        hidden=(h,c)
        
        x=x.permute(0,2,3,1).contiguous().view(shape[0]*shape[2],shape[3],shape[1])
        x,hidden=self.lstm(x,hidden)
        x=x.view(shape[0],shape[2],shape[3],self.c_out).permute(0,3,1,2).contiguous()
        x=self.conv1(x)#b,c,n,l
        return x,hidden[0],hidden[0]



class LSTM_multitemp(nn.Module):
    def __init__(self,device,num_nodes,cluster_nodes,l3cluster_nodes,dropout=0.3,supports=None,supports_cluster=None,l3_supports_cluster=None,transmit=None,l3_transmit=None,length=12, 
                 in_dim=1,in_dim_cluster=3,in_dim_l3=2,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(LSTM_multitemp,self).__init__()


        self.lstm_recent = nn.LSTM(in_dim, dilation_channels, batch_first=True)
        self.lstm_day = nn.LSTM(in_dim, dilation_channels, batch_first=True)
        self.lstm_week = nn.LSTM(in_dim, dilation_channels, batch_first=True)

        self.lstm_recent_cluster = nn.LSTM(in_dim_cluster, dilation_channels, batch_first=True)
        self.lstm_day_cluster = nn.LSTM(in_dim_cluster, dilation_channels, batch_first=True)
        self.lstm_week_cluster = nn.LSTM(in_dim_cluster, dilation_channels, batch_first=True)

        self.lstm_recent_l3 = nn.LSTM(in_dim_l3, dilation_channels, batch_first=True)
        self.lstm_day_l3 = nn.LSTM(in_dim_l3, dilation_channels, batch_first=True)
        self.lstm_week_l3 = nn.LSTM(in_dim_l3, dilation_channels, batch_first=True)
        
        # self.lstm=nn.LSTM(in_dim,dilation_channels,batch_first=True)#b*n,l,c

        self.c_out=dilation_channels
        tem_size=length
        self.tem_size=tem_size
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        
        self.W_recent = nn.Parameter(torch.randn(1, 12, num_nodes, 1).to(device), requires_grad=True)
        self.W_day = nn.Parameter(torch.randn(1, 12, num_nodes, 1).to(device), requires_grad=True)
        self.W_week = nn.Parameter(torch.randn(1, 12, num_nodes, 1).to(device), requires_grad=True)

        
    def forward(self, recent_x, day_x, week_x, recent_x_cluster, day_x_cluster, week_x_cluster, recent_x_cl3, day_x_cl3, week_x_cl3):
        def process_lstm(x, lstm):
            shape = x.shape
            h = Variable(torch.zeros((1, shape[0] * shape[2], self.c_out))).cuda()
            c = Variable(torch.zeros((1, shape[0] * shape[2], self.c_out))).cuda() 
            hidden = (h, c)
            x=x.permute(0,2,3,1).contiguous().view(shape[0]*shape[2],shape[3],shape[1])
            x,hidden=lstm(x,hidden)
            x=x.view(shape[0],shape[2],shape[3],self.c_out).permute(0,3,1,2).contiguous()
            x=self.conv1(x)#b,c,n,l
            return x,hidden[0],hidden[0]
        
        recent_out, _, _ = process_lstm(recent_x, self.lstm_recent)
        day_out, _, _ = process_lstm(day_x, self.lstm_day)
        week_out, _, _ = process_lstm(week_x, self.lstm_week)
        # recent_out_cluster, _, _ = process_lstm(recent_x_cluster, self.lstm_recent_cluster)
        # day_out_cluster, _, _ = process_lstm(day_x_cluster, self.lstm_day_cluster)
        # week_out_cluster, _, _ = process_lstm(week_x_cluster, self.lstm_week_cluster)      
        # recent_out_l3, _, _ = process_lstm(recent_x_cl3, self.lstm_recent_l3)
        # day_out_l3, _, _ = process_lstm(day_x_cl3, self.lstm_day_l3)
        # week_out_l3, _, _ = process_lstm(week_x_cl3, self.lstm_week_l3)
        # print(recent_out.shape)
        # print(self.W_recent.shape)
        x = recent_out * self.W_recent + day_out * self.W_day + week_out * self.W_week
        return x, None, None


class Gated_STGCN(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(Gated_STGCN,self).__init__()
        tem_size=length
        self.block1=ST_BLOCK_4(in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block3=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
    def forward(self,input):
        x=self.bn(input)
        adj=self.supports[0]
        
        x=self.block1(x,adj)
        x=self.block2(x,adj)
        x=self.block3(x,adj)
        x=self.conv1(x)#b,12,n,1
        return x,adj,adj 
    

class HAmodel(nn.Module):
    def __init__(self):
        super(HAmodel, self).__init__()

    def forward(self, input):
        # suppose [batch_size, num_features, num_nodes, time_steps]
        test_x = torch.mean(input[:, :, :, -12:], dim=3)  # [batch_size, num_features, num_nodes]
        test_x = test_x.unsqueeze(-1)  # [batch_size, num_features, num_nodes, 1]
        prediction = test_x.repeat(1, 1, 1, 12)  # [batch_size, num_features, num_nodes, 12]
        return prediction
    

class GRU(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(GRU,self).__init__()
        self.gru=nn.GRU(in_dim,dilation_channels,batch_first=True)#b*n,l,c
        self.c_out=dilation_channels
        tem_size=length
        self.tem_size=tem_size
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1,tem_size),
                          stride=(1,1), bias=True)
        
    def forward(self,input):
        x=input
        shape = x.shape
        h =Variable(torch.zeros((1,shape[0]*shape[2],self.c_out))).cuda()
        hidden=h
        
        x=x.permute(0,2,3,1).contiguous().view(shape[0]*shape[2],shape[3],shape[1])
        x,hidden=self.gru(x,hidden)
        x=x.view(shape[0],shape[2],shape[3],self.c_out).permute(0,3,1,2).contiguous()
        x=self.conv1(x)#b,c,n,l
        return x,hidden[0],hidden[0]


class H_GCN(nn.Module):
    def __init__(self,device, num_nodes, cluster_nodes,dropout=0.3, supports=None,supports_cluster=None,transmit=None,length=12, 
                 in_dim=1,in_dim_cluster=3,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(H_GCN, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        self.transmit=transmit
        self.cluster_nodes=cluster_nodes
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.start_conv_cluster = nn.Conv2d(in_channels=in_dim_cluster,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports
        self.supports_cluster = supports_cluster
        
        self.supports_len = 0
        self.supports_len_cluster = 0
        if supports is not None:
            self.supports_len += len(supports)
            self.supports_len_cluster+=len(supports_cluster)

        
        if supports is None:
            self.supports = []
            self.supports_cluster = []
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_cluster=Parameter(torch.zeros(cluster_nodes,cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)
        self.supports_len +=1
        self.supports_len_cluster +=1
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)  
        self.nodevec1_c = nn.Parameter(torch.randn(cluster_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_c = nn.Parameter(torch.randn(10,cluster_nodes).to(device), requires_grad=True).to(device)  
        
        
        self.block1=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
        self.block2=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
        
        self.block_cluster1=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-6,3,dropout,cluster_nodes,
                            self.supports_len)
        self.block_cluster2=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-9,2,dropout,cluster_nodes,
                            self.supports_len)
        
        self.skip_conv1=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
         
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.conv_cluster1=Conv2d(dilation_channels,out_dim,kernel_size=(1,3),
                          stride=(1,1), bias=True)
        self.bn_cluster=BatchNorm2d(in_dim_cluster,affine=False)
        self.gate1=gate(2*dilation_channels)
        self.gate2=gate(2*dilation_channels)
        self.gate3=gate(2*dilation_channels)
        
        self.transmit1=Transmit(dilation_channels,length,transmit,num_nodes,cluster_nodes)
        self.transmit2=Transmit(dilation_channels,length-6,transmit,num_nodes,cluster_nodes)
        self.transmit3=Transmit(dilation_channels,length-9,transmit,num_nodes,cluster_nodes)
       

    def forward(self, input, input_cluster):
        # print(f"Model input shape: {input.shape}, input_cluster shape: {input_cluster.shape}")
        
        x=self.bn(input)
        shape=x.shape
        input_c=input_cluster
        x_cluster=self.bn_cluster(input_c)
        if self.supports is not None:
            #nodes
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d=1/(torch.sum(A,-1))
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            
            new_supports = self.supports + [A]
            #region
            A_cluster=F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))
            d_c=1/(torch.sum(A_cluster,-1))
            D_c=torch.diag_embed(d_c)
            A_cluster=torch.matmul(D_c,A_cluster)
            
            new_supports_cluster = self.supports_cluster + [A_cluster]
        
        #network
        transmit=self.transmit              
        x = self.start_conv(x)
        x_cluster = self.start_conv_cluster(x_cluster)
        transmit1 = self.transmit1(x,x_cluster)
        x_1=(torch.einsum('bmn,bcnl->bcml',transmit1,x_cluster))   
        
        x=self.gate1(x,x_1)
            
        skip=0
        skip_c=0
        #1
        x_cluster=self.block_cluster1(x_cluster,new_supports_cluster) 
        x=self.block1(x,new_supports)   
        transmit2 = self.transmit2(x,x_cluster)
        x_2=(torch.einsum('bmn,bcnl->bcml',transmit2,x_cluster)) 
        
        x=self.gate2(x,x_2)  
        s1=self.skip_conv1(x)
        skip=s1+skip 

        #2       
        x_cluster=self.block_cluster2(x_cluster,new_supports_cluster)
        x=self.block2(x,new_supports) 
        transmit3 = self.transmit3(x,x_cluster)
        x_3=(torch.einsum('bmn,bcnl->bcml',transmit3,x_cluster)) 
        
        x=self.gate3(x,x_3)  
        
        s2=self.skip_conv2(x)      
        skip = skip[:, :, :,  -s2.size(3):]
        skip = s2 + skip        
        
        #output
        x = F.relu(skip)      
        x = F.relu(self.end_conv_1(x))            
        x = self.end_conv_2(x)              
        return x,transmit3,A
    


class GRCN(nn.Module):      
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(GRCN,self).__init__()
       
        self.block1=ST_BLOCK_5(in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_5(dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        
        self.tem_size=length
        
        self.conv1=Conv2d(dilation_channels,out_dim,kernel_size=(1,length),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
    def forward(self,input):
        x=self.bn(input)
        
        adj=self.supports[0]
        x=self.block1(x,adj)
        x=self.block2(x,adj)
        x=self.conv1(x)
        return x,adj,adj     


class OGCRNN(nn.Module):      
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(OGCRNN,self).__init__()
       
        self.block1=ST_BLOCK_5(in_dim,dilation_channels,num_nodes,length,K,Kt)
        
        self.tem_size=length
        
        self.conv1=Conv2d(dilation_channels,out_dim,kernel_size=(1,length),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
    def forward(self,input):
        x=self.bn(input)
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A=F.dropout(A,0.5)
        x=self.block1(x,A)
        
        x=self.conv1(x)
        return x,A,A   


     
class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1,out_dim=12,residual_channels=32,
                 dilation_channels=32,skip_channels=256,
                 end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        
        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len +=1
            
     
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                
                new_dilation *=2
                receptive_field += additional_scope
                
                additional_scope *= 2
                
                self.gconv.append(multi_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_1=BatchNorm2d(in_dim,affine=False)

    def forward(self, input):
        input=self.bn_1(input)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            x = self.gconv[i](x, new_supports)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)           
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x,adp,adp


# single spatial single temporal granularity
class MGSTGCNsingle(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(MGSTGCNsingle, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.supports = supports
        
        self.supports_len = 0
        
        if supports is not None:
            self.supports_len += len(supports)
        
        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)   
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        
        self.supports_len +=1
        
        Kt1=2
        self.block1=GCNPool(dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
        self.block2=GCNPool(dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
        
        self.skip_conv1=Conv2d(dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        self.bn=BatchNorm2d(in_dim,affine=False)
        

    def forward(self, input):
        x=self.bn(input)
        shape=x.shape
        
        if self.supports is not None:
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d=1/(torch.sum(A,-1))
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            
            new_supports = self.supports + [A]
         
        skip=0
        x = self.start_conv(x)
        
        #1
        x=self.block1(x,new_supports)
        
        s1=self.skip_conv1(x)
        skip=s1+skip
        
        #2
        x=self.block2(x,new_supports)
       
        s2=self.skip_conv2(x)
        skip = skip[:, :, :,  -s2.size(3):]
        skip = s2 + skip
                
        #output
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x,x,A


# multi temporal granularity
class MGSTGCNwms(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(MGSTGCNwms, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.supports = supports
        
        self.supports_len = 0
        
        if supports is not None:
            self.supports_len += len(supports)
        
        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)   
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        
        self.supports_len +=1
        
        Kt1=2
        self.block1=GCNPool(dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
        self.block2=GCNPool(dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
        
        self.skip_conv1=Conv2d(dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        

        self.bn_recent = BatchNorm2d(in_dim, affine=False)
        self.bn_day = BatchNorm2d(in_dim, affine=False)
        self.bn_week = BatchNorm2d(in_dim, affine=False)

        # Add weights for the three components
        self.W_recent = nn.Parameter(torch.randn(1, out_dim, 170, 1).to(device), requires_grad=True)
        self.W_day = nn.Parameter(torch.randn(1, out_dim, 170, 1).to(device), requires_grad=True)
        self.W_week = nn.Parameter(torch.randn(1, out_dim, 170, 1).to(device), requires_grad=True)


    def forward(self, input_recent, input_day, input_week):
        
        # print("input_recent shape:",input_recent.shape)
        # print("input_day shape:",input_day.shape)
        # print("input_week shape:",input_week.shape)

        recent_x = self.bn_recent(input_recent)
        day_x = self.bn_day(input_day)
        week_x = self.bn_week(input_week)
        
        if self.supports is not None:
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d=1/(torch.sum(A,-1))
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            
            new_supports = self.supports + [A]
                 
        skip_recent_x=0
        skip_day_x=0
        skip_week_x=0
        
        recent_x = self.start_conv(recent_x)
        day_x = self.start_conv(day_x)
        week_x = self.start_conv(week_x)
        
        # Process each input through blocks
        recent_x = self.block1(recent_x, new_supports)
        day_x = self.block1(day_x, new_supports)
        week_x = self.block1(week_x, new_supports)

        # Apply skip connections and final convolutions
        s1_recent_x = self.skip_conv1(recent_x)
        skip_recent_x = s1_recent_x + skip_recent_x
        s1_day_x = self.skip_conv1(day_x)
        skip_day_x = s1_day_x + skip_day_x
        s1_week_x = self.skip_conv1(week_x)
        skip_week_x = s1_week_x + skip_week_x

        recent_x = self.block2(recent_x, new_supports)
        day_x = self.block2(day_x, new_supports)
        week_x = self.block2(week_x, new_supports)

        
        s2_recent_x = self.skip_conv2(recent_x)
        skip_recent_x = skip_recent_x[:, :, :, -s2_recent_x.size(3):]
        skip_recent_x = s2_recent_x + skip_recent_x

        s2_day_x = self.skip_conv2(day_x)
        skip_day_x = skip_day_x[:, :, :, -s2_day_x.size(3):]
        skip_day_x = s2_day_x + skip_day_x

        s2_week_x = self.skip_conv2(week_x)
        skip_week_x = skip_week_x[:, :, :, -s2_week_x.size(3):]
        skip_week_x = s2_week_x + skip_week_x
                
        # Output
        x_recent_x = F.relu(skip_recent_x)
        x_recent_x = F.relu(self.end_conv_1(x_recent_x))
        x_recent_x = self.end_conv_2(x_recent_x)

        x_day_x = F.relu(skip_day_x)
        x_day_x = F.relu(self.end_conv_1(x_day_x))
        x_day_x = self.end_conv_2(x_day_x)

        x_week_x = F.relu(skip_week_x)
        x_week_x = F.relu(self.end_conv_1(x_week_x))
        x_week_x = self.end_conv_2(x_week_x)
        
        # Apply weights to each component
        # print("size:",x_recent_x.shape)
        # print("size:",x_day_x.shape)
        # print("size:",x_week_x.shape)
        x_recent_x = x_recent_x * self.W_recent
        x_day_x = x_day_x * self.W_day
        x_week_x = x_week_x * self.W_week

        x = x_recent_x + x_day_x + x_week_x
        return x, x, A 
    

class MGSTGCNwmt(nn.Module):
    def __init__(self,device,num_nodes,cluster_nodes,l3cluster_nodes,dropout=0.3,supports=None,supports_cluster=None,l3_supports_cluster=None,transmit=None,l3_transmit=None,length=12, 
                 in_dim=1,in_dim_cluster=3,in_dim_l3=2,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(MGSTGCNwmt, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        self.transmit=transmit
        self.l3_transmit=l3_transmit
        self.cluster_nodes=cluster_nodes
        self.l3cluster_nodes=l3cluster_nodes

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.start_conv_cluster = nn.Conv2d(in_channels=in_dim_cluster,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.start_conv_l3 = nn.Conv2d(in_channels=in_dim_l3,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        self.supports = supports
        self.supports_cluster = supports_cluster
        self.l3_supports_cluster = l3_supports_cluster
        
        self.supports_len = 0
        self.supports_len_cluster = 0
        self.supports_len_l3 = 0

        if supports is not None:
            self.supports_len += len(supports)
            self.supports_len_cluster+=len(supports_cluster)
            self.supports_len_l3 += len(l3_supports_cluster)

        
        if supports is None:
            self.supports = []
            self.supports_cluster = []
            self.l3_supports_cluster = []
        
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_cluster=Parameter(torch.zeros(cluster_nodes,cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)
        self.h_l3 = Parameter(torch.zeros(l3cluster_nodes, l3cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_l3, a=0, b=0.0001)
        self.supports_len +=1
        self.supports_len_cluster +=1
        self.supports_len_l3 += 1
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)  
        self.nodevec1_c = nn.Parameter(torch.randn(cluster_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_c = nn.Parameter(torch.randn(10,cluster_nodes).to(device), requires_grad=True).to(device)  
        self.nodevec1_l3 = nn.Parameter(torch.randn(l3cluster_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_l3 = nn.Parameter(torch.randn(10, l3cluster_nodes).to(device), requires_grad=True).to(device)
        
    # S-T Block 1
        self.block1=GCNPool(3*dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
    # S-T Block 2
        self.block2=GCNPool(3*dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
    # S-T Block 3    
        self.block_cluster1=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-6,3,dropout,cluster_nodes,
                            self.supports_len)
    # S-T Block 4
        self.block_cluster2=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-9,2,dropout,cluster_nodes,
                            self.supports_len)
        
    # S-T Block 5
        self.block_l3_1 = GCNPool(dilation_channels,dilation_channels,l3cluster_nodes,length-6,3,dropout,l3cluster_nodes,
                            self.supports_len)
    # S-T Block 6
        self.block_l3_2 = GCNPool(dilation_channels,dilation_channels,l3cluster_nodes,length-9,2,dropout,l3cluster_nodes,
                            self.supports_len)
        
        self.skip_conv1=Conv2d(3*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(3*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.conv_cluster1=Conv2d(dilation_channels,out_dim,kernel_size=(1,3),
                          stride=(1,1), bias=True)
        self.bn_cluster=BatchNorm2d(in_dim_cluster,affine=False)
        # l3
        self.bn_l3 = BatchNorm2d(in_dim_l3, affine=False)


        self.gate1=gate(2*dilation_channels)
        self.gate2=gate(2*dilation_channels)
        self.gate3=gate(2*dilation_channels)

        self.gate4=gatel3(2*dilation_channels)
        self.gate5=gatel3(2*dilation_channels)
        self.gate6=gatel3(2*dilation_channels)

        self.adjust_channels = nn.Conv2d(64, 32, kernel_size=1)  # 新增调整通道数的卷积层
        self.transmit4 = Transmit(dilation_channels,length,l3_transmit,num_nodes,l3cluster_nodes)
        self.transmit5 = Transmit(dilation_channels,length-6,l3_transmit,num_nodes,l3cluster_nodes)
        self.transmit6 = Transmit(dilation_channels,length-9,l3_transmit,num_nodes,l3cluster_nodes)
        self.transmit1=Transmit(dilation_channels,length,transmit,num_nodes,cluster_nodes)
        self.transmit2=Transmit(dilation_channels,length-6,transmit,num_nodes,cluster_nodes)
        self.transmit3=Transmit(dilation_channels,length-9,transmit,num_nodes,cluster_nodes)
       
    def forward(self, input, input_cluster, input_cl3):
        # print("input_cluster shape:",input_cluster.shape)
        # print("input_cl3 shape:",input_cl3.shape)
        x=self.bn(input)
        x_cluster=self.bn_cluster(input_cluster)
        # l3
        x_cl3 = self.bn_l3(input_cl3)

        if self.supports is not None:
            #nodes
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d=1/(torch.sum(A,-1))
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            new_supports = self.supports + [A]
            
            #region
            A_cluster=F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))
            d_c=1/(torch.sum(A_cluster,-1))
            D_c=torch.diag_embed(d_c)
            A_cluster=torch.matmul(D_c,A_cluster)
            new_supports_cluster = self.supports_cluster + [A_cluster]

            # L3 Cluster
            A_l3 = F.relu(torch.mm(self.nodevec1_l3, self.nodevec2_l3))
            d_l3 = 1 / (torch.sum(A_l3, -1))
            D_l3 = torch.diag_embed(d_l3)
            A_l3 = torch.matmul(D_l3, A_l3)
            new_supports_l3 = self.l3_supports_cluster + [A_l3]
        
        # Linear Transformation             
        x = self.start_conv(x)
        x_cluster = self.start_conv_cluster(x_cluster)
        x_cl3 = self.start_conv_l3(x_cl3)
        
        # print("x shape:", x.shape)
        # print("x_cluster shape:", x_cluster.shape)
        # print("x_cl3 shape:", x_cl3.shape)
      
        # transmit1 = self.transmit1(x,x_cluster)
        # x_1=(torch.einsum('bmn,bcnl->bcml',transmit1,x_cluster))   
        # x=self.gate1(x,x_1)
        # # print("Shape of x_cluster after block1:", x_cluster.shape)

        transmit4 = self.transmit4(x, x_cl3)
        x_4 = torch.einsum('bmn,bcnl->bcml', transmit4, x_cl3)
        
        transmit1 = self.transmit1(x, x_cluster)
        x_1 = torch.einsum('bmn,bcnl->bcml', transmit1, x_cluster)
       

        x = self.gate4(x, x_1, x_4)


        skip=0
        skip_c=0
        
        #1
        # S-T Block 5
        x_cl3=self.block_l3_1(x_cl3,new_supports_l3)
        # S-T Block 3
        x_cluster=self.block_cluster1(x_cluster,new_supports_cluster)
        # S-T Block 1 
        x=self.block1(x,new_supports)
        
        transmit5 = self.transmit5(x, x_cl3)
        x_5 = torch.einsum('bmn,bcnl->bcml', transmit5, x_cl3)
        
        transmit2 = self.transmit2(x, x_cluster)
        x_2 = torch.einsum('bmn,bcnl->bcml', transmit2, x_cluster)
        
        x = self.gate5(x, x_5, x_2)
        s1=self.skip_conv1(x)
        skip=s1+skip 

        #2
        # S-T Block 6
        x_cl3=self.block_l3_2(x_cl3,new_supports_l3)
        # S-T Block 4
        x_cluster=self.block_cluster2(x_cluster,new_supports_cluster)
        # S-T Block 2
        x=self.block2(x,new_supports)         
        transmit6 = self.transmit6(x, x_cl3)
        x_6 = torch.einsum('bmn,bcnl->bcml', transmit6, x_cl3)
        
        transmit3 = self.transmit3(x, x_cluster)
        x_3 = torch.einsum('bmn,bcnl->bcml', transmit3, x_cluster)
       
        x = self.gate6(x, x_6, x_3)
        s2=self.skip_conv2(x)      
        skip = skip[:, :, :,  -s2.size(3):]
        skip = s2 + skip       
        
        #output
        x = F.relu(skip)      
        x = F.relu(self.end_conv_1(x))            
        x = self.end_conv_2(x)              
        return x,transmit3,transmit6,A    



class MultGran_STGCN(nn.Module):
    def __init__(self,device,num_nodes,cluster_nodes,l3cluster_nodes,dropout=0.3,supports=None,supports_cluster=None,l3_supports_cluster=None,transmit=None,l3_transmit=None,length=12, 
                 in_dim=1,in_dim_cluster=3,in_dim_l3=2,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(MultGran_STGCN, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        self.transmit=transmit
        self.l3_transmit=l3_transmit
        self.cluster_nodes=cluster_nodes
        self.l3cluster_nodes=l3cluster_nodes

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.start_conv_cluster = nn.Conv2d(in_channels=in_dim_cluster,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.start_conv_l3 = nn.Conv2d(in_channels=in_dim_l3,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        self.supports = supports
        self.supports_cluster = supports_cluster
        self.l3_supports_cluster = l3_supports_cluster
        
        self.supports_len = 0
        self.supports_len_cluster = 0
        self.supports_len_l3 = 0

        if supports is not None:
            self.supports_len += len(supports)
            self.supports_len_cluster+=len(supports_cluster)
            self.supports_len_l3 += len(l3_supports_cluster)

        
        if supports is None:
            self.supports = []
            self.supports_cluster = []
            self.l3_supports_cluster = []
        
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_cluster=Parameter(torch.zeros(cluster_nodes,cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)
        self.h_l3 = Parameter(torch.zeros(l3cluster_nodes, l3cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_l3, a=0, b=0.0001)
        self.supports_len +=1
        self.supports_len_cluster +=1
        self.supports_len_l3 += 1

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)  
        self.nodevec1_c = nn.Parameter(torch.randn(cluster_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_c = nn.Parameter(torch.randn(10,cluster_nodes).to(device), requires_grad=True).to(device)  
        self.nodevec1_l3 = nn.Parameter(torch.randn(l3cluster_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_l3 = nn.Parameter(torch.randn(10, l3cluster_nodes).to(device), requires_grad=True).to(device)
        
    # S-T Block 1
        self.block1=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
    # S-T Block 2
        self.block2=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
    # S-T Block 3    
        self.block_cluster1=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-6,3,dropout,cluster_nodes,
                            self.supports_len)
    # S-T Block 4
        self.block_cluster2=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-9,2,dropout,cluster_nodes,
                            self.supports_len)
        
    # S-T Block 5
        self.block_l3_1 = GCNPool(dilation_channels,dilation_channels,l3cluster_nodes,length-6,3,dropout,l3cluster_nodes,
                            self.supports_len)
    # S-T Block 6
        self.block_l3_2 = GCNPool(dilation_channels,dilation_channels,l3cluster_nodes,length-9,2,dropout,l3cluster_nodes,
                            self.supports_len)
    
        self.block3=GCNPool(dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
        self.block4=GCNPool(dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
        
        self.block7=GCNPool(3*dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
        self.block8=GCNPool(3*dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
        
        self.skip_conv1=Conv2d(3*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(3*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        self.skip_conv3=Conv2d(dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv4=Conv2d(dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        
        self.conv_cluster1=Conv2d(dilation_channels,out_dim,kernel_size=(1,3),
                          stride=(1,1), bias=True)
        
        
        self.bn_recent = BatchNorm2d(in_dim, affine=False)
        self.bn_day = BatchNorm2d(in_dim, affine=False)
        self.bn_week = BatchNorm2d(in_dim, affine=False)

        self.bn_cluster_recent = BatchNorm2d(in_dim_cluster, affine=False)
        self.bn_cluster_day = BatchNorm2d(in_dim_cluster, affine=False)
        self.bn_cluster_week = BatchNorm2d(in_dim_cluster, affine=False)

        self.bn_l3_recent = BatchNorm2d(in_dim_l3, affine=False)
        self.bn_l3_day = BatchNorm2d(in_dim_l3, affine=False)
        self.bn_l3_week = BatchNorm2d(in_dim_l3, affine=False)
        

        self.gate1=gate(2*dilation_channels)
        self.gate2=gate(2*dilation_channels)
        self.gate3=gate(2*dilation_channels)

        self.gate4=gatel3(2*dilation_channels)
        self.gate5=gatel3(2*dilation_channels)
        self.gate6=gatel3(2*dilation_channels)

        self.adjust_channels = nn.Conv2d(96, 64, kernel_size=1)

        self.W_recent = nn.Parameter(torch.randn(1, out_dim, num_nodes, 1).to(device), requires_grad=True)
        self.W_day = nn.Parameter(torch.randn(1, out_dim, num_nodes, 1).to(device), requires_grad=True)
        self.W_week = nn.Parameter(torch.randn(1, out_dim, num_nodes, 1).to(device), requires_grad=True)

        self.transmit4 = Transmit(dilation_channels,length,l3_transmit,num_nodes,l3cluster_nodes)
        self.transmit5 = Transmit(dilation_channels,length-6,l3_transmit,num_nodes,l3cluster_nodes)
        self.transmit6 = Transmit(dilation_channels,length-9,l3_transmit,num_nodes,l3cluster_nodes)
        self.transmit1=Transmit(dilation_channels,length,transmit,num_nodes,cluster_nodes)
        self.transmit2=Transmit(dilation_channels,length-6,transmit,num_nodes,cluster_nodes)
        self.transmit3=Transmit(dilation_channels,length-9,transmit,num_nodes,cluster_nodes)
       

    def forward(self, recent_x, day_x, week_x, recent_x_cluster, day_x_cluster, week_x_cluster, recent_x_cl3, day_x_cl3, week_x_cl3):
        recent_x = self.bn_recent(recent_x)
        day_x = self.bn_day(day_x)
        week_x = self.bn_week(week_x)
        recent_x_cluster = self.bn_cluster_recent(recent_x_cluster)
        
        recent_x_cl3 = self.bn_l3_recent(recent_x_cl3)
        
        if self.supports is not None:
            #nodes
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d=1/(torch.sum(A,-1))
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            new_supports = self.supports + [A]
            
            #region
            A_cluster=F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))
            d_c=1/(torch.sum(A_cluster,-1))
            D_c=torch.diag_embed(d_c)
            A_cluster=torch.matmul(D_c,A_cluster)
            new_supports_cluster = self.supports_cluster + [A_cluster]

            # L3 Cluster
            A_l3 = F.relu(torch.mm(self.nodevec1_l3, self.nodevec2_l3))
            d_l3 = 1 / (torch.sum(A_l3, -1))
            D_l3 = torch.diag_embed(d_l3)
            A_l3 = torch.matmul(D_l3, A_l3)
            new_supports_l3 = self.l3_supports_cluster + [A_l3]

        # Linear Transformation             
        recent_x = self.start_conv(recent_x)
        day_x = self.start_conv(day_x)
        week_x = self.start_conv(week_x)
        recent_x_cluster = self.start_conv_cluster(recent_x_cluster)
        
        recent_x_cl3 = self.start_conv_l3(recent_x_cl3)
        
        transmit4_recent = self.transmit4(recent_x, recent_x_cl3)
        x_4_recent = torch.einsum('bmn,bcnl->bcml', transmit4_recent, recent_x_cl3)
        

        transmit1_recent = self.transmit1(recent_x, recent_x_cluster)
        x_1_recent = torch.einsum('bmn,bcnl->bcml', transmit1_recent, recent_x_cluster)
       

        recent_x = self.gate4(recent_x, x_1_recent, x_4_recent)
        
        skip_recent = 0
        skip_day = 0
        skip_week = 0
     
        #1
        # S-T Block 5
        recent_x_cl3 = self.block_l3_1(recent_x_cl3, new_supports_l3)
        
        # S-T Block 3
        recent_x_cluster = self.block_cluster1(recent_x_cluster, new_supports_cluster)
        
        # S-T Block 1 
        recent_x = self.block7(recent_x, new_supports)
        
        day_x = self.block3(day_x, new_supports)
        week_x = self.block3(week_x, new_supports)

        transmit5_recent = self.transmit5(recent_x, recent_x_cl3)
        x_5_recent = torch.einsum('bmn,bcnl->bcml', transmit5_recent, recent_x_cl3)
        
        transmit2_recent = self.transmit2(recent_x, recent_x_cluster)
        x_2_recent = torch.einsum('bmn,bcnl->bcml', transmit2_recent, recent_x_cluster)
        
        recent_x = self.gate5(recent_x, x_5_recent, x_2_recent)
        

        s1_recent = self.skip_conv1(recent_x)
        skip_recent = s1_recent + skip_recent

        s1_day = self.skip_conv3(day_x)
        skip_day = s1_day + skip_day

        s1_week = self.skip_conv3(week_x)
        skip_week = s1_week + skip_week
        
       
        #2
        # S-T Block 6
        recent_x_cl3 = self.block_l3_2(recent_x_cl3, new_supports_l3)
        
        # S-T Block 4
        recent_x_cluster = self.block_cluster2(recent_x_cluster, new_supports_cluster)
        
        # S-T Block 2
        recent_x = self.block8(recent_x, new_supports)
        
        day_x = self.block4(day_x, new_supports)
        week_x = self.block4(week_x, new_supports)

        transmit6_recent = self.transmit6(recent_x, recent_x_cl3)
        x_6_recent = torch.einsum('bmn,bcnl->bcml', transmit6_recent, recent_x_cl3)
        
        transmit3_recent = self.transmit3(recent_x, recent_x_cluster)
        x_3_recent = torch.einsum('bmn,bcnl->bcml', transmit3_recent, recent_x_cluster)
       
        recent_x = self.gate6(recent_x, x_6_recent, x_3_recent)
        
        
        s2_recent = self.skip_conv2(recent_x)
        skip_recent = skip_recent[:, :, :, -s2_recent.size(3):]
        skip_recent = s2_recent + skip_recent

        s2_day = self.skip_conv4(day_x)
        skip_day = skip_day[:, :, :, -s2_day.size(3):]
        skip_day = s2_day + skip_day

        s2_week = self.skip_conv4(week_x)
        skip_week = skip_week[:, :, :, -s2_week.size(3):]
        skip_week = s2_week + skip_week        
        
        # Output
        x_recent = F.relu(skip_recent)
        x_recent = F.relu(self.end_conv_1(x_recent))
        x_recent = self.end_conv_2(x_recent)

        x_day = F.relu(skip_day)
        x_day = F.relu(self.end_conv_1(x_day))
        x_day = self.end_conv_2(x_day)

        x_week = F.relu(skip_week)
        x_week = F.relu(self.end_conv_1(x_week))
        x_week = self.end_conv_2(x_week)

        # Apply weights to each component
        x_recent = x_recent * self.W_recent
        x_day = x_day * self.W_day
        x_week = x_week * self.W_week

        x = x_recent + x_day + x_week             
        return x,transmit3_recent,transmit6_recent,A