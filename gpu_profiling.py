import time
import argparse
import numpy as np
import torch
import torch.nn as nn


def test_mm(device):
    torch.manual_seed(1007)
    torch.cuda.manual_seed(1007)
    matrix_size = 4096
    n_warmup = 3
    n_test = 10
    # Set up a simple operation to time, e.g., matrix multiplication
    a = torch.randn(matrix_size, matrix_size).to(device)
    b = torch.randn(matrix_size, matrix_size).to(device)
    
    # Warm-up the GPU (optional, helps with more accurate timing)
    for _ in range(n_warmup):
        _ = torch.mm(a, b)
    
    # Synchronize before starting
    if device==torch.device("cuda"):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    
    # Start the timing
    if device==torch.device("cuda"):
        start_event.record()
    elif device==torch.device("mps") or device==torch.device("cpu"):
        start_event=time.time()
    
    result_sum = 0
    for _ in range(n_test):
        result = torch.mm(a, b)  # Replace with your desired GPU job
        result_sum = result_sum + result
    
    # Waits for everything to finish running
    if device==torch.device("cuda"):
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
    elif device==torch.device("mps") or device==torch.device("cpu"):
        end_event=time.time()
        elapsed_time_ms = (end_event - start_event)*1000
        
    return elapsed_time_ms/1000, result_sum


def build_mlp(input_dim, output_dim, hiddens):
    n_neurons = [input_dim] + hiddens + [output_dim]
    layers = []
    for i in range(len(n_neurons)-1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
        layers.append(torch.nn.ReLU())
    del layers[-1]
    return nn.Sequential(*layers)

def test_mlp(device):
    import torch.nn
    torch.manual_seed(1007)
    torch.cuda.manual_seed(1007)
    
    n_batchsize = 8192 * 16
    n_warmup = 2
    n_test = 3
    input_dim = 256
    output_dim = 1
    
    model = build_mlp(input_dim, output_dim, [256, 256, 256, 256]).to(device)
    inputs = torch.randn(n_batchsize, input_dim).to(device)
    results_sum = 0
    if device==torch.device("cuda"):
        torch.cuda.synchronize()
    for i in range(n_warmup+n_test):
        if i==n_warmup:
            t_start = time.time()
        with torch.no_grad():
            results = model(inputs)
            results_sum = results_sum + results
    if device==torch.device("cuda"):
        torch.cuda.synchronize()
    t_end = time.time()
    elapsed_time = t_end - t_start
    # print("mean(results)=",torch.mean(results))
    return elapsed_time, results_sum

def test_cnn(device):
    import torchvision.models as models
    torch.manual_seed(1007)
    torch.cuda.manual_seed(1007)
    n_batchsize = 8
    n_warmup = 5
    n_test = 10
    
    results_sum = 0
    model = models.resnet18().to(device)
    inputs = torch.randn(n_batchsize, 3, 224, 224).to(device)
    if device==torch.device("cuda"):
        torch.cuda.synchronize()
    for i in range(n_warmup+n_test):
        if i==n_warmup:
            t_start = time.time()
        with torch.no_grad():
            results = model(inputs)
            results_sum = results_sum + results
    if device==torch.device("cuda"):
        torch.cuda.synchronize()
    t_end = time.time()
    elapsed_time = t_end - t_start
    # print("mean(results)=",torch.mean(results))
    return elapsed_time, results_sum

def test_gnn(device):
    from torch.nn import Linear, ReLU, Dropout
    from torch_geometric.data import Data, Batch
    from torch_geometric.datasets import FakeDataset
    from torch_geometric.nn import Sequential, GCNConv
    torch.manual_seed(1007)
    torch.cuda.manual_seed(1007)
    
    num_graphs = 2048
    in_channels = 32
    out_channels = 1
    n_warmup = 5
    n_test = 10
    
    model = Sequential('x, edge_index', [
        (GCNConv(in_channels, 256), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(256, 256), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(256, 256), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(256, 256), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(256, 64), 'x, edge_index -> x'),
        ReLU(inplace=True),
        Linear(64, out_channels),
    ]).to(device)
    
    fake_dataset = FakeDataset(num_graphs=num_graphs, avg_num_nodes=6, avg_degree=3, num_channels=in_channels)
    data = fake_dataset[0:num_graphs]._data.to(device)
    results_sum = 0
    if device==torch.device("cuda"):
        torch.cuda.synchronize()
    for i in range(n_warmup+n_test):
        if i==n_warmup:
            t_start = time.time()
        with torch.no_grad():
            results = model(data.x, data.edge_index)
            results_sum = results_sum + results
    if device==torch.device("cuda"):
        torch.cuda.synchronize()
    t_end = time.time()
    elapsed_time = t_end - t_start
    # print("mean(results)=",torch.mean(results))
    return elapsed_time, results_sum


def main():
    # Check if GPU is available
    if args.device=="mps":
        device = torch.device("mps")
    elif args.device=="gpu":
        device = torch.device("cuda")
    elif args.device=="cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    runtime_d = {}
    for test_mode in args.tests:
        print(f"Run {test_mode} test...")
        if test_mode == "mm":
            runtime_d[test_mode], res = test_mm(device)
        elif test_mode == "mlp":
            runtime_d[test_mode], res = test_mlp(device)
        elif test_mode == "cnn":
            runtime_d[test_mode], res = test_cnn(device)
        elif test_mode == "gnn":
            runtime_d[test_mode], res = test_gnn(device)
        else:
            raise NotImplementedError
    print("-"*20)
    for test_mode in args.tests:
        print(f"{test_mode:5s}: {runtime_d[test_mode]:.7f} s")
    return runtime_d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "mps", "default"], default="default")
    parser.add_argument("--tests", type=str, nargs="+", default=["mm", "mlp", "cnn", "gnn"])
    parser.add_argument("--n_trials", type=int, default=5) 
    args = parser.parse_args()

    ttt1=time.time()
    runtime_d_list={}
    for trial_i in range(args.n_trials):
        runtime_d = main()
        for key in runtime_d:
            if key not in runtime_d_list:
                runtime_d_list[key]=[]
            runtime_d_list[key].append(runtime_d[key])
    
    print("="*20)
    for test_mode in args.tests:
        print(f"{test_mode:5s}: {np.mean(runtime_d_list[test_mode]):.7f} ± {np.std(runtime_d_list[test_mode]):.7f} s")
    ttt2=time.time()
    print("Finished in %.3f seconds"%(ttt2 - ttt1))