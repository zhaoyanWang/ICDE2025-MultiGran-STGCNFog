
# Note that we initially constructed a three-dimensional DP table for scheduling, however since for each device we only got one, 
# therefore no need to be constrained by device number per type of device. So in the paper manuscript, we simplied it into a two-dimensional dp table, 
# which was test before the 3D one.

import json
import networkx as nx
import random
import math
import numpy as np
import random

class Device:
    def __init__(self, name, processing_time_per_layer, bandwidth, memory, num_instances):
        self.name = name
        self.processing_time_per_layer = processing_time_per_layer
        self.bandwidth = bandwidth  # Mbps
        self.memory = memory  # MB
        self.num_instances = num_instances

class Layer:
    def __init__(self, mem_MB, input_params, output_params):
        self.mem_MB = mem_MB
        self.input_params = input_params
        self.output_params = output_params

def compute_standard_deviation(times):
    return np.std(times)

def compute_time(t_comp, layer_l, layer_r):
    return sum(t_comp[layer_l - 1:layer_r])

def layer_bytes_out(layers, layer, dtype_size, batch_size):
    num_elements = sum(layers[layer - 1].output_params.values())
    return num_elements * dtype_size * batch_size

def comm_time(layers, layer_r, batch_size, dtype_size, device_u, device_v):
    dat_bytes = layer_bytes_out(layers, layer_r, dtype_size, batch_size)
    mbits_sec = min(device_u.bandwidth, device_v.bandwidth)
    bytes_sec = mbits_sec * 1024 * 1024 / 8
    return dat_bytes / bytes_sec

def is_layers_fit(device, layers, layer_l, layer_r, dtype_size, batch_size, data_buffers_in, data_buffers_out):
    mem_required = sum(layer.mem_MB for layer in layers[layer_l - 1:layer_r]) * 1024 * 1024
    
    if layer_l == 1:
        mem_required += sum(layers[layer_l - 1].input_params.values()) * dtype_size * batch_size
    else:
        mem_required += layer_bytes_out(layers, layer_l - 1, dtype_size, batch_size) * data_buffers_in

    mem_required += layer_bytes_out(layers, layer_r, dtype_size, batch_size) * data_buffers_out
    mem_required += layer_bytes_out(layers, layer_r, dtype_size, batch_size)

    return device.memory * 1024 * 1024 >= mem_required

def schedule_layers_to_devices(layers, devices, batch_size, dtype_size, data_buffers_in, data_buffers_out):
    num_layers = len(layers)
    num_devices = len(devices)

    device_counts = [device.num_instances for device in devices]
    mask = 1
    for count in device_counts:
        mask *= count + 1

    dp = [[[float('inf')] * num_devices for _ in range(mask)] for _ in range(num_layers + 1)]
    parent = [[[(-1, -1)] * num_devices for _ in range(mask)] for _ in range(num_layers + 1)]

    prefix_product = [1] * (len(device_counts) + 1)
    for i in range(1, len(prefix_product)):
        prefix_product[i] = prefix_product[i - 1] * (device_counts[i - 1] + 1)

    for u in range(num_devices):
        dp[0][0][u] = 0
    
    for i in range(num_layers):
        for S in range(mask):
            for u in range(num_devices):
                if dp[i][S][u] == float('inf'):
                    continue
                
                if S // prefix_product[u] % (device_counts[u] + 1) == device_counts[u]:
                    continue

                for j in range(i + 1, num_layers + 1):
                    if not is_layers_fit(devices[u], layers, i + 1, j, dtype_size, batch_size, data_buffers_in, data_buffers_out):
                        continue
                    
                    computation_time = compute_time(devices[u].processing_time_per_layer, i + 1, j)
                    for v in range(num_devices):
                        communication_time = 0
                        if u != v:
                            communication_time = comm_time(layers, j, batch_size, dtype_size, devices[u], devices[v])
                            wait_time = 0
                        wait_time = max(0, (dp[i][S][u] + communication_time) - dp[i][S][v])                       
                        cost = max(dp[i][S][u], max(computation_time, communication_time)) 
            
                        S_new = S + prefix_product[u]
                        if cost < dp[j][S_new][v]:
                            dp[j][S_new][v] = cost
                            parent[j][S_new][v] = (i, u)
    
    best_cost = float('inf')
    best_j, best_S, best_u = -1, -1, -1
    for S in range(mask):
        for u in range(num_devices):
            if dp[num_layers][S][u] < best_cost:
                best_cost = dp[num_layers][S][u]
                best_j, best_S, best_u = num_layers, S, u

    best_schedule = []
    j, S, u = best_j, best_S, best_u
    while j > 0:
        i, prev_u = parent[j][S][u]
        best_schedule.append((devices[prev_u].name, i + 1, j))
        S -= prefix_product[prev_u]
        j, u = i, prev_u
    
    best_schedule.reverse()

    return best_schedule, best_cost


def compute_cost_and_schedule(layers, devices, layer_order, batch_size, dtype_size, data_buffers_in, data_buffers_out):
    sorted_layers = [layers[i] for i in layer_order]
    sorted_devices = []
    for device in devices:
        sorted_processing_time = [device.processing_time_per_layer[i] for i in layer_order]
        sorted_device = Device(device.name, sorted_processing_time, device.bandwidth, device.memory, device.num_instances)
        sorted_devices.append(sorted_device)

    schedule, cost = schedule_layers_to_devices(sorted_layers, sorted_devices, batch_size, dtype_size, data_buffers_in, data_buffers_out)
    
    device_times = {device_name: 0 for device_name, _, _ in schedule}

    for device_name, start, end in schedule:
        device_idx = next(i for i, d in enumerate(sorted_devices) if d.name == device_name)
        
        processing_time = compute_time(sorted_devices[device_idx].processing_time_per_layer, start, end)
        
        device_times[device_name] += processing_time
    
    device_times_list = [device_times[device_name] for device_name, _, _ in schedule]
    
    load_balance_std = compute_standard_deviation(device_times_list)
    # print("device",device_times_list)
    # print("load_balance_std",load_balance_std)
    return schedule, cost, load_balance_std, device_times_list



def fast_non_dominated_sort(population):
    fronts = [[]]
    for i, individual1 in enumerate(population):
        S = []
        n = 0
        for j, individual2 in enumerate(population):
            if dominates(individual1['objectives'], individual2['objectives']):
                S.append(j)
            elif dominates(individual2['objectives'], individual1['objectives']):
                n += 1
        individual1['dominated'] = S
        individual1['domination_count'] = n
        if n == 0:
            individual1['rank'] = 0
            fronts[0].append(i)
    
    i = 0
    while fronts[i]:
        next_front = []
        for individual_index in fronts[i]:
            for dominated_index in population[individual_index]['dominated']:
                population[dominated_index]['domination_count'] -= 1
                if population[dominated_index]['domination_count'] == 0:
                    population[dominated_index]['rank'] = i + 1
                    next_front.append(dominated_index)
        i += 1
        fronts.append(next_front)
    
    return fronts[:-1]

def calculate_crowding_distance(front, population):
    if len(front) == 0:
        return
    distances = [0] * len(front)
    num_objectives = len(population[0]['objectives'])
    for m in range(num_objectives):
        front.sort(key=lambda x: population[x]['objectives'][m])
        distances[0] = distances[-1] = float('inf')
        for i in range(1, len(front) - 1):
            distances[i] += (population[front[i + 1]]['objectives'][m] - population[front[i - 1]]['objectives'][m])
    
    for i in range(len(front)):
        population[front[i]]['crowding_distance'] = distances[i]

def dominates(objectives1, objectives2):
    return all(x <= y for x, y in zip(objectives1, objectives2)) and any(x < y for x, y in zip(objectives1, objectives2))

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child1 = [None] * size
    child1[start:end] = parent1[start:end]
    
    child2 = [None] * size
    child2[start:end] = parent2[start:end]
    
    pos = end
    for gene in parent2:
        if gene not in child1:
            if pos >= size:
                pos = 0
            child1[pos] = gene
            pos += 1
    
    pos = end
    for gene in parent1:
        if gene not in child2:
            if pos >= size:
                pos = 0
            child2[pos] = gene
            pos += 1

    return child1, child2

def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(individual)), 2))
        individual[i], individual[j] = individual[j], individual[i]

def is_valid_schedule(individual, G):
    position = {node: i for i, node in enumerate(individual)}
    for node in G.nodes:
        for pred in G.predecessors(node):
            if position[pred] > position[node]:
                return False
    return True

def repair_schedule(individual, G):
    sorted_individual = list(nx.topological_sort(G))
    valid_individual = []
    
    for node in individual:
        if node in sorted_individual:
            valid_individual.append(node)
            sorted_individual.remove(node)
    
    valid_individual.extend(sorted_individual)
    
    return valid_individual

def genetic_algorithm_schedule_optimization(G, layers, devices, num_generations, population_size, mutation_rate, crossover_rate, batch_size, dtype_size, data_buffers_in, data_buffers_out):
    population = []
    for _ in range(population_size):
        individual = list(nx.topological_sort(G))
        random.shuffle(individual)
        population.append({
            'individual': individual,
            'objectives': None,
            'dominated': None,
            'domination_count': None,
            'rank': None,
            'crowding_distance': 0
        })

    def fitness(individual):
        schedule, cost, std, _ = compute_cost_and_schedule(layers, devices, individual, batch_size, dtype_size, data_buffers_in, data_buffers_out)
        return [cost, std], schedule, cost, std

    for generation in range(num_generations):
        for individual in population:
            if individual['objectives'] is None:
                individual['objectives'], schedule, cost, std = fitness(individual['individual'])
        
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            calculate_crowding_distance(front, population)
        
        new_population = []
        for front in fronts:
            front.sort(key=lambda x: population[x]['crowding_distance'], reverse=True)
            new_population.extend([population[i] for i in front])
            if len(new_population) >= population_size:
                new_population = new_population[:population_size]
                break
        
        offspring = []
        while len(offspring) < population_size:
            parent1, parent2 = random.sample(new_population, 2)
            
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1['individual'], parent2['individual'])
            else:
                child1, child2 = parent1['individual'], parent2['individual']
            
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            
            if not is_valid_schedule(child1, G):
                child1 = repair_schedule(child1, G)
            if not is_valid_schedule(child2, G):
                child2 = repair_schedule(child2, G)
            
            offspring.append({
                'individual': child1,
                'objectives': None,
                'dominated': None,
                'domination_count': None,
                'rank': None,
                'crowding_distance': 0
            })
            offspring.append({
                'individual': child2,
                'objectives': None,
                'dominated': None,
                'domination_count': None,
                'rank': None,
                'crowding_distance': 0
            })
        
        population = new_population + offspring

        valid_population = [ind for ind in population if ind['objectives'] is not None]
        
        print(f"Generation {generation}: Best Cost = {min(ind['objectives'][0] for ind in valid_population)}")

    best_individual = min(valid_population, key=lambda ind: (ind['rank'], -ind['crowding_distance']))

    best_sort = best_individual['individual']
    _, best_schedule, best_cost, std = fitness(best_sort)

    return best_sort, best_schedule, best_cost




def schedule_ga(G, layers, devices, num_generations=50, population_size=20, mutation_rate=0.1, crossover_rate=0.8, batch_size=16, dtype_size=4):
    data_buffers_in = 1
    data_buffers_out = 1
    
    best_sort, best_schedule, best_cost = genetic_algorithm_schedule_optimization(
        G, layers, devices, num_generations, population_size, mutation_rate, crossover_rate, batch_size, dtype_size, data_buffers_in, data_buffers_out
    )
    
    return best_sort, best_schedule, best_cost

def schedule():
    dependencies = {
        0: [],
        1: [0, 10, 13],
        2: [1, 11, 14],
        3: [2, 12, 15],
        4: [],
        5: [4],
        6: [5],
        7: [],
        8: [7],
        9: [8],
        10: [],
        11: [10],
        12: [11],
        13: [],
        14: [13],
        15: [14],
        16: [3],
        17: [6],
        18: [9],
        19: [16, 17, 18]
    }

    G = nx.DiGraph()

    for layer, deps in dependencies.items():
        G.add_node(layer)
        for dep in deps:
            G.add_edge(dep, layer)

    with open('model_data.json', 'r') as f:
        model_data = json.load(f)

    with open('devices.json', 'r') as f:
        devices_data = json.load(f)

    # dtype_size for torch.float32
    dtype_size = 4

    layers = []
    for i in range(len(model_data["memory_per_layer"])):
        layer = Layer(
            mem_MB=model_data['memory_per_layer'][i],
            input_params=model_data['input_parameters_per_layer'][i],
            output_params=model_data['output_parameters_per_layer'][i]
        )
        layers.append(layer)

    devices = []
    for device_name, data in devices_data.items():
        processing_times = data["processing_time_per_layer"]
        device = Device(
            name=device_name,
            processing_time_per_layer=processing_times,
            bandwidth=data["system_bandwidth_capability"],
            memory=data["system_memory_capacity"],
            num_instances=data["num_instances"]
        )
        devices.append(device)

    # print("Layer Objects:")
    # for idx, layer in enumerate(layers):
    #     print(f"Layer {idx + 1}: Memory: {layer.mem_MB} MB, Input Params: {layer.input_params}, Output Params: {layer.output_params}")

    # print("\nDevice Objects:")
    # for device in devices:
    #     print(f"Device Name: {device.name}, Processing Times: {device.processing_time_per_layer}, Bandwidth: {device.bandwidth} Mbps, Memory: {device.memory} MB")
    
    data_buffers_in = model_data.get("data_buffers_in", 1)
    data_buffers_out = model_data.get("data_buffers_out", 1)

    batch_size = 16
    best_sort, best_schedule, best_cost = schedule_ga(G, layers, devices)

    return best_sort, best_schedule, best_cost
