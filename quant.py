import torch.nn as nn
import torch
import time 
import os
import csv

class ModelQuantization(nn.Module):
    def __init__(self,model):
        super(ModelQuantization,self).__init__()
        
        # Inserting a quantization operator before the input
        self.quantstub = torch.quantization.QuantStub()
        
        # Inserting a dequantization operator after the output
        self.dequantstub = torch.quantization.DeQuantStub()
        
        # Original floating point model
        self.model = model

    def forward(self, x):
        # Floating values to integer values
        z = self.quantstub(x)
        
        z = self.model(z)
        
        # Integer values to floating values
        z = self.dequantstub(z)
        return z
    
def model_calibration(model, data, device):
    
    model = model.to(device)
    model.eval()
    
    for input_values, target_values in data:
        input_values,target_values = input_values.to(device), target_values.to(device)
        
        _ = model(input_values)
        
def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave



def save_model(model, model_name,cpu_latency,gpu_latency,accuracy, is_quantized):
    # specify the path and name of the saved model
    if is_quantized:
        model_name = f"{model_name}_quantized.pth"
    else:
        model_name = f"{model_name}.pth"
    model_path = f"/home/sourav/research_project/{model_name}"

    # save the model
    torch.save(model.state_dict(), model_path)

    # get the file size of the model in bytes
    file_size = os.path.getsize(model_path)

    # convert the file size to MB
    file_size_mb = file_size / (1024 * 1024)

    # print the name, quantization status, and file size in MB
    print(f"Saved model: {model_name} (Quantized: {is_quantized}), File size: {file_size_mb:.2f} MB")
    
    

def save_model_and_stats(model, model_name, cpu_latency, accuracy, is_quantized):
    # specify the path and name of the saved model
    if is_quantized:
        model_name = f"{model_name}_quantized.pth"
    else:
        model_name = f"{model_name}.pth"
    model_path = f"/home/sourav/research_project/{model_name}"

    # save the model
    torch.save(model.state_dict(), model_path)

    # get the file size of the model in bytes
    file_size = os.path.getsize(model_path)

    # convert the file size to MB
    file_size_mb = file_size / (1024 * 1024)

    # print the name, quantization status, and file size in MB
    print(f"Saved model: {model_name} (Quantized: {is_quantized}), File size: {file_size_mb:.2f} MB")

    stats_path = "/home/sourav/research_project/stats.csv"

    
    # check if the stats.csv file exists, create it if it doesn't exist
    if not os.path.exists(stats_path):
        with open(stats_path, mode="w", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Model Name', 'File Size (MB)', 'CPU Latency (ms/sample)', 'Accuracy'])

    # append the architecture, model size, latencies, and accuracy to the csv file
    with open(stats_path, mode="a", newline='') as csv_file:
        writer = csv.writer(csv_file)
        model_name_with_quantized = model_name 
        writer.writerow([model_name_with_quantized, round(file_size_mb, 2), cpu_latency, accuracy])

def save_quantized_model(model, model_name, cpu_latency, gpu_latency, accuracy):
    # specify the path and name of the saved model
    model_name = f"{model_name}_quantized.pth"
    model_path = f"/home/sourav/research_project/saved_models/{model_name}"

    # save the quantized model
    torch.save(model.state_dict(), model_path)

    # get the file size of the model in bytes
    file_size = os.path.getsize(model_path)

    # convert the file size to MB
    file_size_mb = file_size / (1024 * 1024)

    # print the name, quantization status, and file size in MB
    print(f"Saved model: {model_name} (Quantized), File size: {file_size_mb:.2f} MB")


def measure_latency(model, input_shape):
    input_data = torch.randn(input_shape)
    model.eval()  # set model to evaluation mode

    # Warm up the model by running it once
    with torch.no_grad():
        _ = model(input_data)

    # Measure the inference time
    start_time = time.time()
    with torch.no_grad():
        _ = model(input_data)
    end_time = time.time()

    latency = end_time - start_time
    print(f"Inference latency: {latency:.6f} seconds")
    return latency