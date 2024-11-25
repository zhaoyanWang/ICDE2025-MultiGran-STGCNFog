"""Spatial temporal GCN model."""
import torch.nn.functional as F
import torch
from torch import nn
from utils import GCNPool
from utils import Transmit
from utils import gatel3
from torch.nn import BatchNorm2d, Conv2d, Parameter
import torch.multiprocessing as mp
import time
import gc
import psutil
import os


class MultiGran_STGCNLayerShard(nn.Module):
    """Shard for MultiGran_STGCN model."""
    def __init__(self, config):
        super(MultiGran_STGCNLayerShard, self).__init__()
        # config
        self.config = config
        self.dropout = config['dropout']
        self.num_nodes = config['num_nodes']
        self.cluster_nodes = config['cluster_nodes']
        self.l3cluster_nodes = config['l3cluster_nodes']
        self.transmit = config['transmit']
        self.l3_transmit = config['l3_transmit']
        self.supports = config.get('supports', [])
        self.supports_cluster = config.get('supports_cluster', [])
        self.l3_supports_cluster = config.get('l3_supports_cluster', [])
        self.supports_len = len(self.supports)
        self.supports_len_cluster = len(self.supports_cluster)
        self.supports_len_l3 = len(self.l3_supports_cluster)

        if self.supports is None:
            self.supports = []
        if self.supports_cluster is None:
            self.supports_cluster = []
        if self.l3_supports_cluster is None:
            self.l3_supports_cluster = []

        self.h = Parameter(torch.zeros(self.num_nodes, self.num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)

        self.h_cluster = Parameter(torch.zeros(self.cluster_nodes, self.cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)

        self.h_l3 = Parameter(torch.zeros(self.l3cluster_nodes, self.l3cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_l3, a=0, b=0.0001)

        self.supports_len += 1
        self.supports_len_cluster += 1
        self.supports_len_l3 += 1
        self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(config['device']), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(config['device']), requires_grad=True)
        self.nodevec1_c = nn.Parameter(torch.randn(self.cluster_nodes, 10).to(config['device']), requires_grad=True)
        self.nodevec2_c = nn.Parameter(torch.randn(10, self.cluster_nodes).to(config['device']), requires_grad=True)
        self.nodevec1_l3 = nn.Parameter(torch.randn(self.l3cluster_nodes, 10).to(config['device']), requires_grad=True)
        self.nodevec2_l3 = nn.Parameter(torch.randn(10, self.l3cluster_nodes).to(config['device']), requires_grad=True)

        self._build_shard()
        self.recent_x = None
        self.day_x = None 
        self.week_x = None
        self.recent_x_cluster = None
        self.recent_x_cl3 = None
        
        self.new_supports = None
        self.new_supports_cluster = None
        self.new_supports_l3 = None



    def has_layer(self, layer_idx):
        """Check if a layer is present in the configuration."""
        return layer_idx in self.config['layers']


    # def measure_layer_performance(self, layer_idx):
    #     """Measure performance metrics for a specific layer."""
    #     process = psutil.Process(os.getpid())

    #     gc.collect()
    #     layer_start_mem = process.memory_info().rss / 1000000  # Start memory in MB
    #     layer_start_time = time.time()

    #     with torch.no_grad():
    #         self.forward_single_layer(layer_idx)

    #     layer_end_time = time.time()
    #     gc.collect()
    #     layer_end_mem = process.memory_info().rss / 1000000  # End memory in MB

    #     memory_usage = layer_end_mem - layer_start_mem
    #     time_taken = layer_end_time - layer_start_time

    #     input_params = sum(p.numel() for p in self.parameters())  # Number of input parameters
    #     output_params = sum(p.numel() for p in self.parameters())  # Number of output parameters

    #     return memory_usage, time_taken, input_params, output_params

    def _profile_layer(self, layer_idx, inputs):
        process = psutil.Process(os.getpid())
        gc.collect()
        start_mem = process.memory_info().rss
        start_time = time.time()
        outputs, prediction, dimensions, input_params, output_params = self.forward_single_layer(layer_idx, inputs)
        end_time = time.time()
        gc.collect()
        end_mem = process.memory_info().rss
        memory_usage = end_mem - start_mem
        time_taken = end_time - start_time

        return memory_usage, time_taken, outputs, prediction, dimensions, input_params, output_params
    
    def measure_layer_performance(self, layer_idx, inputs):
        """Measure performance metrics for a specific layer using a separate process."""
        queue = mp.Queue()
        proc = mp.Process(target=self._profile_layer_process, args=(queue, layer_idx, inputs))
        proc.start()

        mem_usage, time_taken, outputs, prediction, dimensions, input_params, output_params = queue.get()

        proc.join()

        return mem_usage, time_taken, outputs, prediction, dimensions, input_params, output_params
    
    def _profile_layer_process(self, queue, layer_idx, inputs):
        mem_usage, time_taken, outputs, prediction, dimensions, input_params, output_params = self._profile_layer(layer_idx, inputs)
        
        # Detach the tensors in outputs to remove the gradient information
        detached_outputs = {key: val.detach() if torch.is_tensor(val) else val for key, val in outputs.items()}
        if torch.is_tensor(prediction):
            prediction = prediction.detach()
        queue.put((mem_usage, time_taken, detached_outputs, prediction, dimensions, input_params, output_params))

    def _build_shard(self):

        if self.has_layer(0):
            self.bn_recent = BatchNorm2d(self.config['in_dim'], affine=False)
            self.start_conv = nn.Conv2d(
                in_channels=self.config['in_dim'],
                out_channels=self.config['residual_channels'],
                kernel_size=(1, 1)
            )
            
        if self.has_layer(1):
            self.transmit4 = Transmit(
                self.config['dilation_channels'], 
                self.config['length'], 
                self.config['l3_transmit'], 
                self.num_nodes, 
                self.l3cluster_nodes
            )
            self.transmit1 = Transmit(
                self.config['dilation_channels'], 
                self.config['length'], 
                self.config['transmit'], 
                self.num_nodes, 
                self.cluster_nodes
            )
            self.gate4 = gatel3(2 * self.config['dilation_channels'])    
            self.block7 = GCNPool(
                3 * self.config['dilation_channels'],
                self.config['dilation_channels'],
                self.num_nodes,
                self.config['length'] - 6,
                3,
                self.dropout,
                self.num_nodes,
                self.supports_len
            )

        if self.has_layer(2):
            self.transmit5 = Transmit(
                self.config['dilation_channels'],
                self.config['length'] - 6,
                self.config['l3_transmit'],
                self.num_nodes,
                self.l3cluster_nodes
            )
            self.transmit2 = Transmit(
                self.config['dilation_channels'],
                self.config['length'] - 6,
                self.config['transmit'],
                self.num_nodes,
                self.cluster_nodes
            )
            self.gate5 = gatel3(2 * self.config['dilation_channels'])
            self.block8 = GCNPool(
                3 * self.config['dilation_channels'],
                self.config['dilation_channels'],
                self.num_nodes,
                self.config['length'] - 9,
                2,
                self.dropout,
                self.num_nodes,
                self.supports_len
            )

        if self.has_layer(3):
            self.transmit6 = Transmit(
                self.config['dilation_channels'],
                self.config['length'] - 9,
                self.config['l3_transmit'],
                self.num_nodes,
                self.l3cluster_nodes
            )
            self.transmit3 = Transmit(
                self.config['dilation_channels'],
                self.config['length'] - 9,
                self.config['transmit'],
                self.num_nodes,
                self.cluster_nodes
            )
            self.gate6 = gatel3(2 * self.config['dilation_channels'])
            self.skip_conv1 = Conv2d(
                3 * self.config['dilation_channels'],
                self.config['skip_channels'],
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=True
            )
            self.skip_conv2 = Conv2d(
                3 * self.config['dilation_channels'],
                self.config['skip_channels'],
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=True
            )

        if self.has_layer(4):
            self.bn_day = BatchNorm2d(self.config['in_dim'], affine=False)
            self.start_conv = nn.Conv2d(
                in_channels=self.config['in_dim'],
                out_channels=self.config['residual_channels'],
                kernel_size=(1, 1)
            )

        if self.has_layer(5):
            self.block3 = GCNPool(
                self.config['dilation_channels'],
                self.config['dilation_channels'],
                self.num_nodes,
                self.config['length'] - 6,
                3,
                self.dropout,
                self.num_nodes,
                self.supports_len
            )

        if self.has_layer(6):
            self.skip_conv3 = Conv2d(
                self.config['dilation_channels'],
                self.config['skip_channels'],
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=True
            )
            self.block4 = GCNPool(
                self.config['dilation_channels'],
                self.config['dilation_channels'],
                self.num_nodes,
                self.config['length'] - 9,
                2,
                self.dropout,
                self.num_nodes,
                self.supports_len
            )
            self.skip_conv4 = Conv2d(
                self.config['dilation_channels'],
                self.config['skip_channels'],
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=True
            )

        if self.has_layer(7):
            self.bn_week = BatchNorm2d(self.config['in_dim'], affine=False)
            self.start_conv = nn.Conv2d(
                in_channels=self.config['in_dim'],
                out_channels=self.config['residual_channels'],
                kernel_size=(1, 1)
            )

        if self.has_layer(8):
            self.block3 = GCNPool(
                self.config['dilation_channels'],
                self.config['dilation_channels'],
                self.num_nodes,
                self.config['length'] - 6,
                3,
                self.dropout,
                self.num_nodes,
                self.supports_len
            )

        if self.has_layer(9):
            self.skip_conv3 = Conv2d(
                self.config['dilation_channels'],
                self.config['skip_channels'],
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=True
            )
            self.block4 = GCNPool(
                self.config['dilation_channels'],
                self.config['dilation_channels'],
                self.num_nodes,
                self.config['length'] - 9,
                2,
                self.dropout,
                self.num_nodes,
                self.supports_len
            )
            self.skip_conv4 = Conv2d(
                self.config['dilation_channels'],
                self.config['skip_channels'],
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=True
            )

        if self.has_layer(10):
            self.bn_cluster_recent = BatchNorm2d(self.config['in_dim_cluster'], affine=False)
            self.start_conv_cluster = nn.Conv2d(
                in_channels=self.config['in_dim_cluster'],
                out_channels=self.config['residual_channels'],
                kernel_size=(1, 1)
            )

        if self.has_layer(11):
            self.block_cluster1 = GCNPool(
                self.config['dilation_channels'],
                self.config['dilation_channels'],
                self.cluster_nodes,
                self.config['length'] - 6,
                3,
                self.config['dropout'],
                self.cluster_nodes,
                self.supports_len_cluster
            )

        if self.has_layer(12):
            self.block_cluster2 = GCNPool(
                self.config['dilation_channels'],
                self.config['dilation_channels'],
                self.cluster_nodes,
                self.config['length'] - 9,
                2,
                self.config['dropout'],
                self.cluster_nodes,
                self.supports_len_cluster
            )

        if self.has_layer(13):
            self.bn_l3_recent = BatchNorm2d(self.config['in_dim_l3'], affine=False)
            self.start_conv_l3 = nn.Conv2d(
                in_channels=self.config['in_dim_l3'],
                out_channels=self.config['residual_channels'],
                kernel_size=(1, 1)
            )

        if self.has_layer(14):
            self.block_l3_1 = GCNPool(
                self.config['dilation_channels'],
                self.config['dilation_channels'],
                self.l3cluster_nodes,
                self.config['length'] - 6,
                3,
                self.config['dropout'],
                self.l3cluster_nodes,
                self.supports_len_l3
            )

        if self.has_layer(15):
            self.block_l3_2 = GCNPool(
                self.config['dilation_channels'],
                self.config['dilation_channels'],
                self.l3cluster_nodes,
                self.config['length'] - 9,
                2,
                self.config['dropout'],
                self.l3cluster_nodes,
                self.supports_len_l3
            )

        if self.has_layer(16):
            self.end_conv_1 = nn.Conv2d(
                in_channels=self.config['skip_channels'],
                out_channels=self.config['end_channels'],
                kernel_size=(1, 3),
                bias=True
            )
            self.end_conv_2 = nn.Conv2d(
                in_channels=self.config['end_channels'],
                out_channels=self.config['out_dim'],
                kernel_size=(1, 1),
                bias=True
            )
            self.W_recent = nn.Parameter(
                torch.randn(1, self.config['out_dim'], self.num_nodes, 1).to(self.config['device']),
                requires_grad=True
            )
            self.W_day = nn.Parameter(
                torch.randn(1, self.config['out_dim'], self.num_nodes, 1).to(self.config['device']),
                requires_grad=True
            )
            self.W_week = nn.Parameter(
                torch.randn(1, self.config['out_dim'], self.num_nodes, 1).to(self.config['device']),
                requires_grad=True
            )

    
    def forward_single_layer(self, layer_idx, inputs):
        """Process inputs through a specific layer based on the layer index."""
        outputs = {}
        dimensions = {}
        input_params = {}
        output_params = {}
        result = None
        # print("layer:",layer_idx)
        
        if self.has_layer(0) and layer_idx == 0:
                dimensions['layer0_input_from_input_recent'] = self.recent_x.size()

                input_params['layer0_input_from_input_recent'] = self.recent_x.numel()
                
                recent_x = self.bn_recent(self.recent_x)    
                recent_x = self.start_conv(recent_x)
                
                dimensions['layer0_output_to_layer1'] = recent_x.size()
                output_params['layer0_output_to_layer1'] = recent_x.numel()
                outputs['layer0'] = recent_x


        if self.has_layer(1) and layer_idx == 1:
            recent_x = inputs['layer0']
            recent_x_cluster_from_layer10 = inputs['layer10_DTB']
            recent_x_cl3_from_layer13 = inputs['layer13_DTB']
            
            dimensions['layer1_input_from_layer0'] = recent_x.size()
            dimensions['layer1_input_from_layer10'] = recent_x_cluster_from_layer10.size()
            dimensions['layer1_input_from_layer13'] = recent_x_cl3_from_layer13.size()

            input_params['layer1_input_from_layer0'] = recent_x.numel()
            input_params['layer1_input_from_layer10'] = recent_x_cluster_from_layer10.numel()
            input_params['layer1_input_from_layer13'] = recent_x_cl3_from_layer13.numel()
            
            # recent_x DTB1 + ST block 1
            transmit4_recent = self.transmit4(recent_x, recent_x_cl3_from_layer13)
            x_4_recent = torch.einsum('bmn,bcnl->bcml', transmit4_recent, recent_x_cl3_from_layer13)
        
            transmit1_recent = self.transmit1(recent_x, recent_x_cluster_from_layer10)
            x_1_recent = torch.einsum('bmn,bcnl->bcml', transmit1_recent, recent_x_cluster_from_layer10)
        
            recent_x = self.gate4(recent_x, x_1_recent, x_4_recent)
            recent_x = self.block7(recent_x, self.new_supports)
            
            dimensions['layer1_output_to_layer2'] = recent_x.size()
            output_params['layer1_output_to_layer2'] = recent_x.numel()
            outputs['layer1'] = recent_x


        if self.has_layer(2) and layer_idx == 2:
            recent_x = inputs['layer1']
            recent_x_cluster_from_layer11 = inputs['layer11_DTB']
            recent_x_cl3_from_layer14 = inputs['layer14_DTB']

            dimensions['layer2_input_from_layer1'] = recent_x.size()
            dimensions['layer2_input_from_layer11'] = recent_x_cluster_from_layer11.size()
            dimensions['layer2_input_from_layer14'] = recent_x_cl3_from_layer14.size()

            input_params['layer2_input_from_layer1'] = recent_x.numel()
            input_params['layer2_input_from_layer11'] = recent_x_cluster_from_layer11.numel()
            input_params['layer2_input_from_layer14'] = recent_x_cl3_from_layer14.numel()

            # recent_x DTB2 + ST block 2
            transmit5_recent = self.transmit5(recent_x, recent_x_cl3_from_layer14)
            x_5_recent = torch.einsum('bmn,bcnl->bcml', transmit5_recent, recent_x_cl3_from_layer14)
            
            transmit2_recent = self.transmit2(recent_x, recent_x_cluster_from_layer11)
            x_2_recent = torch.einsum('bmn,bcnl->bcml', transmit2_recent, recent_x_cluster_from_layer11)
            
            recent_x = self.gate5(recent_x, x_5_recent, x_2_recent)
            
            
            outputs['layer2_skip'] = recent_x
            dimensions['layer2_output_to_layer3_skip'] = recent_x.size()
            output_params['layer2_output_to_layer3_skip'] = recent_x.numel()

            recent_x = self.block8(recent_x, self.new_supports)

            dimensions['layer2_output_to_layer3'] = recent_x.size()
            output_params['layer2_output_to_layer3'] = recent_x.numel()
            outputs['layer2'] = recent_x


        if self.has_layer(3) and layer_idx == 3:
            recent_x = inputs['layer2']
            recent_x_from_layer2 = inputs['layer2_skip']
            recent_x_cl3_from_layer15 = inputs['layer15_DTB']
            recent_x_cluster_from_layer12 = inputs['layer12_DTB']

            dimensions['layer3_input_from_layer2_skip'] = recent_x_from_layer2.size()
            dimensions['layer3_input_from_layer2'] = recent_x.size()
            dimensions['layer3_input_from_layer12'] = recent_x_cluster_from_layer12.size()
            dimensions['layer3_input_from_layer15'] = recent_x_cl3_from_layer15.size()

            input_params['layer3_input_from_layer2_skip'] = recent_x_from_layer2.numel()
            input_params['layer3_input_from_layer2'] = recent_x.numel()
            input_params['layer3_input_from_layer12'] = recent_x_cluster_from_layer12.numel()
            input_params['layer3_input_from_layer15'] = recent_x_cl3_from_layer15.numel()
            
            transmit6_recent = self.transmit6(recent_x, recent_x_cl3_from_layer15)
            x_6_recent = torch.einsum('bmn,bcnl->bcml', transmit6_recent, recent_x_cl3_from_layer15)
            transmit3_recent = self.transmit3(recent_x, recent_x_cluster_from_layer12)
            x_3_recent = torch.einsum('bmn,bcnl->bcml', transmit3_recent, recent_x_cluster_from_layer12)
            recent_x = self.gate6(recent_x, x_6_recent, x_3_recent)

            s1_recent = self.skip_conv1(recent_x_from_layer2)
            skip_recent = 0
            skip_recent = s1_recent + skip_recent

            s2_recent = self.skip_conv2(recent_x)
            skip_recent = skip_recent[:, :, :, -s2_recent.size(3):]
            skip_recent = s2_recent + skip_recent
            
            dimensions['layer3_output_to_layer16'] = skip_recent.size()
            output_params['layer3_output_to_layer16'] = skip_recent.numel()
            outputs['layer3'] = skip_recent
            

        if self.has_layer(4) and layer_idx == 4:
            dimensions['layer4_input_from_input_daily'] = self.day_x.size()
            input_params['layer4_input_from_input_daily'] = self.day_x.numel()

            day_x = self.bn_day(self.day_x)
            day_x = self.start_conv(day_x)

            dimensions['layer4_output_to_layer5'] = day_x.size()
            output_params['layer4_output_to_layer5'] = day_x.numel()
            outputs['layer4'] = day_x


        if self.has_layer(5) and layer_idx == 5:
            day_x = inputs['layer4']
            dimensions['layer5_input_from_layer4'] = day_x.size()
            input_params['layer5_input_from_layer4'] = day_x.numel()

            day_x = self.block3(day_x, self.new_supports)
            dimensions['layer5_output_to_layer6'] = day_x.size()
            output_params['layer5_output_to_layer6'] = day_x.numel()
            outputs['layer5'] = day_x


        if self.has_layer(6) and layer_idx == 6:
            day_x = inputs['layer5']
            dimensions['layer6_input_from_layer5'] = day_x.size()
            input_params['layer6_input_from_layer5'] = day_x.numel()

            skip_day = 0
            s1_day = self.skip_conv3(day_x)

            # daily_x ST block 2 + Skip_conv
            day_x = self.block4(day_x, self.new_supports)

            skip_day = s1_day + skip_day
            s2_day = self.skip_conv4(day_x)
            skip_day = skip_day[:, :, :, -s2_day.size(3):]
            skip_day = s2_day + skip_day

            dimensions['layer6_output_to_layer17'] = skip_day.size()
            output_params['layer6_output_to_layer17'] = skip_day.numel()
            outputs['layer6'] = skip_day

        if self.has_layer(7) and layer_idx == 7:
            dimensions['layer7_input_from_input_weekly'] = self.week_x.size()
            input_params['layer7_input_from_input_weekly'] = self.week_x.numel()

            week_x = self.bn_week(self.week_x)
            week_x = self.start_conv(week_x)

            dimensions['layer7_output_to_layer8'] = week_x.size()
            output_params['layer7_output_to_layer8'] = week_x.numel()
            outputs['layer7'] = week_x

        if self.has_layer(8) and layer_idx == 8:
            week_x = inputs['layer7']
            dimensions['layer8_input_from_layer7'] = week_x.size()
            input_params['layer8_input_from_layer7'] = week_x.numel()

            week_x = self.block3(week_x, self.new_supports)

            dimensions['layer8_output_to_layer9'] = week_x.size()
            output_params['layer8_output_to_layer9'] = week_x.numel()
            outputs['layer8'] = week_x

        if self.has_layer(9) and layer_idx == 9:
            week_x = inputs['layer8']
            dimensions['layer9_input_from_layer8'] = week_x.size()
            input_params['layer9_input_from_layer8'] = week_x.numel()

            skip_week = 0
            s1_week = self.skip_conv3(week_x)
            week_x = self.block4(week_x, self.new_supports)

            skip_week = s1_week + skip_week
            s2_week = self.skip_conv4(week_x)
            skip_week = skip_week[:, :, :, -s2_week.size(3):]
            skip_week = s2_week + skip_week

            dimensions['layer9_output_to_layer18'] = skip_week.size()
            output_params['layer9_output_to_layer18'] = skip_week.numel()
            outputs['layer9'] = skip_week

        if self.has_layer(10) and layer_idx == 10:
            dimensions['layer10_input_from_input_recentcluster'] = self.recent_x_cluster.size()
            input_params['layer10_input_from_input_recentcluster'] = self.recent_x_cluster.numel()

            recent_x_cluster = self.bn_cluster_recent(self.recent_x_cluster)
            recent_x_cluster = self.start_conv_cluster(recent_x_cluster)
            
            dimensions['layer10_output_to_layer1'] = recent_x_cluster.size()
            dimensions['layer10_output_to_layer11'] = recent_x_cluster.size()
            output_params['layer10_output_to_layer1'] = recent_x_cluster.numel()
            output_params['layer10_output_to_layer11'] = recent_x_cluster.numel()
            outputs['layer10'] = recent_x_cluster
            outputs['layer10_DTB'] = recent_x_cluster
            

        if self.has_layer(11) and layer_idx == 11:
            recent_x_cluster = inputs['layer10']
            dimensions['layer11_input_from_layer10'] = recent_x_cluster.size()
            input_params['layer11_input_from_layer10'] = recent_x_cluster.numel()

            recent_x_cluster = self.block_cluster1(recent_x_cluster, self.new_supports_cluster)
            
            dimensions['layer11_output_to_layer2'] = recent_x_cluster.size()
            dimensions['layer11_output_to_layer12'] = recent_x_cluster.size()
            output_params['layer11_output_to_layer2'] = recent_x_cluster.numel()
            output_params['layer11_output_to_layer12'] = recent_x_cluster.numel()

            outputs['layer11'] = recent_x_cluster
            outputs['layer11_DTB'] = recent_x_cluster

        if self.has_layer(12) and layer_idx == 12:
            recent_x_cluster = inputs['layer11']
            dimensions['layer12_input_from_layer11'] = recent_x_cluster.size()
            input_params['layer12_input_from_layer11'] = recent_x_cluster.numel()

            recent_x_cluster = self.block_cluster2(recent_x_cluster, self.new_supports_cluster)

            dimensions['layer12_output_to_layer3'] = recent_x_cluster.size()
            output_params['layer12_output_to_layer3'] = recent_x_cluster.numel()

            outputs['layer12_DTB'] = recent_x_cluster

        if self.has_layer(13) and layer_idx == 13:
            dimensions['layer13_input_from_input_recentl3'] = self.recent_x_cl3.size()
            input_params['layer13_input_from_input_recentl3'] = self.recent_x_cl3.numel()

            recent_x_cl3 = self.bn_l3_recent(self.recent_x_cl3)
            recent_x_cl3 = self.start_conv_l3(recent_x_cl3)

            dimensions['layer13_output_to_layer1'] = recent_x_cl3.size()
            dimensions['layer13_output_to_layer14'] = recent_x_cl3.size()
            output_params['layer13_output_to_layer1'] = recent_x_cl3.numel()
            output_params['layer13_output_to_layer14'] = recent_x_cl3.numel()

            outputs['layer13'] = recent_x_cl3
            outputs['layer13_DTB'] = recent_x_cl3

        if self.has_layer(14) and layer_idx == 14:
            recent_x_cl3 = inputs['layer13']
            dimensions['layer14_input_from_layer13'] = recent_x_cl3.size()
            input_params['layer14_input_from_layer13'] = recent_x_cl3.numel()

            recent_x_cl3 = self.block_l3_1(recent_x_cl3, self.new_supports_l3)

            dimensions['layer14_output_to_layer2'] = recent_x_cl3.size()
            dimensions['layer14_output_to_layer15'] = recent_x_cl3.size()
            output_params['layer14_output_to_layer2'] = recent_x_cl3.numel()
            output_params['layer14_output_to_layer15'] = recent_x_cl3.numel()

            outputs['layer14'] = recent_x_cl3
            outputs['layer14_DTB'] = recent_x_cl3

        if self.has_layer(15) and layer_idx == 15:
            recent_x_cl3 = inputs['layer14']
            dimensions['layer15_input_from_layer14'] = recent_x_cl3.size()
            input_params['layer15_input_from_layer14'] = recent_x_cl3.numel()

            recent_x_cl3 = self.block_l3_2(recent_x_cl3, self.new_supports_l3)

            dimensions['layer15_output_to_layer3'] = recent_x_cl3.size()
            output_params['layer15_output_to_layer3'] = recent_x_cl3.numel()

            outputs['layer15_DTB'] = recent_x_cl3


        if self.has_layer(16) and layer_idx == 16:
            skip_recent_from_layer3 = inputs['layer3']

            dimensions['layer16_input_from_layer3'] = skip_recent_from_layer3.size()
            input_params['layer16_input_from_layer3'] = skip_recent_from_layer3.numel()
            
            x_recent = F.relu(skip_recent_from_layer3)
            x_recent = F.relu(self.end_conv_1(x_recent))
            x_recent = self.end_conv_2(x_recent)

            dimensions['layer16_output_to_layer19'] = x_recent.size()
            output_params['layer16'] = x_recent.numel()

            outputs['layer16'] = x_recent


        if self.has_layer(17) and layer_idx == 17:
            skip_day_from_layer6 = inputs['layer6']
            dimensions['layer17_input_from_layer6'] = skip_day_from_layer6.size()
            input_params['layer17_input_from_layer6'] = skip_day_from_layer6.numel()

            x_day = F.relu(skip_day_from_layer6)
            x_day = F.relu(self.end_conv_1(x_day))
            x_day = self.end_conv_2(x_day)

            dimensions['layer17_output_to_layer19'] = x_day.size()
            output_params['layer17'] = x_day.numel()

            outputs['layer17'] = x_day
        

        if self.has_layer(18) and layer_idx == 18:
            skip_week_from_layer9 = inputs['layer9']
            dimensions['layer18_input_from_layer9'] = skip_week_from_layer9.size()
            input_params['layer18_input_from_layer9'] = skip_week_from_layer9.numel()

            x_week = F.relu(skip_week_from_layer9)
            x_week = F.relu(self.end_conv_1(x_week))
            x_week = self.end_conv_2(x_week)

            dimensions['layer18_output_to_layer19'] = x_week.size()
            output_params['layer18'] = x_week.numel()

            outputs['layer18'] = x_week



        if self.has_layer(19) and layer_idx == 19:
            result_recent = inputs['layer16']
            result_day = inputs['layer17']
            result_week = inputs['layer18']

            dimensions['layer19_input_from_layer16'] = result_recent.size()
            dimensions['layer19_input_from_layer17'] = result_day.size()
            dimensions['layer19_input_from_layer18'] = result_week.size()

            input_params['layer19_input_from_layer16'] = result_recent.numel()
            input_params['layer19_input_from_layer17'] = result_day.numel()
            input_params['layer19_input_from_layer18'] = result_week.numel()

            # Apply weights to each component
            x_recent = result_recent * self.W_recent
            x_day = result_day * self.W_day
            x_week = result_week * self.W_week

            result = x_recent + x_day + x_week
            dimensions['layer19_final_result'] = result.size()
            output_params['layer19_final_result'] = result.numel()

        return outputs, result, dimensions, input_params, output_params
    

    @torch.no_grad()
    def forward(self, recent_x, day_x, week_x, recent_x_cluster, day_x_cluster, week_x_cluster, recent_x_cl3, day_x_cl3, week_x_cl3):
        mem_MB = []
        parameters_in = []
        parameters_out = []
        time_s = []
        final_dimensions = {}
        inputs = {
            'recent_x': recent_x,
            'day_x': day_x,
            'week_x': week_x,
            'recent_x_cluster': recent_x_cluster,
            'recent_x_cl3': recent_x_cl3
        }
        
        if self.supports is not None:
            #nodes
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d=1/(torch.sum(A,-1))
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            self.new_supports = self.supports + [A]
            
            #region
            A_cluster=F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))
            d_c=1/(torch.sum(A_cluster,-1))
            D_c=torch.diag_embed(d_c)
            A_cluster=torch.matmul(D_c,A_cluster)
            self.new_supports_cluster = self.supports_cluster + [A_cluster]

            # L3 Cluster
            A_l3 = F.relu(torch.mm(self.nodevec1_l3, self.nodevec2_l3))
            d_l3 = 1 / (torch.sum(A_l3, -1))
            D_l3 = torch.diag_embed(d_l3)
            A_l3 = torch.matmul(D_l3, A_l3)
            self.new_supports_l3 = self.l3_supports_cluster + [A_l3]

        self.recent_x = recent_x
        self.day_x = day_x 
        self.week_x = week_x
        self.recent_x_cluster = recent_x_cluster
        self.recent_x_cl3 = recent_x_cl3

        for layer_idx in [0, 10, 13, 1, 11, 14, 2, 12, 15, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19]:
            mem_usage, time_taken, outputs, prediction, dimensions, input_params, output_params = self.measure_layer_performance(layer_idx, inputs)

            mem_MB.append(mem_usage)
            time_s.append(time_taken)
            inputs.update(outputs)
            parameters_in.append(input_params)
            parameters_out.append(output_params)
            final_dimensions.update(dimensions)
                
        
        print(final_dimensions)
        print(f"Memory per layer (MB): {mem_MB}")
        print(f"Input parameters per layer: {parameters_in}")
        print(f"Output parameters per layer: {parameters_out}")
        print(f"Processing time per layer (s): {time_s}")

        return prediction,None,None,None