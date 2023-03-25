import torch.nn as nn
import torch
import torch.optim as optim
from torch import Tensor
from typing import Type
import torchvision
import torchvision.transforms as transforms
import time
import copy
import os
import numpy as np


import json
import os
import argparse


import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import matplotlib.pyplot as plt


from loader import *
from models import MODELS_MAP
from misc import *
from quant import *

def validate(val_loader, model, criterion, device):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            


# ################
parser = argparse.ArgumentParser(description='Quantization Experiments')
parser.add_argument(
    '--config', default='config.json', type=str, help='config file')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)
config_experiment_number = config['experiment_number']
config_dataset = config['dataset']
config_architecture = config['architecture']
config_batch_size = config['batch_size']
config_lr = config['lr']
config_momentum = config['momentum']
config_random_seed = config['random_seed']
config_weight_decay = config['weight_decay']
config_backend = config['backend']
config_epoch = config['epoch']


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu_dev = torch.device('cpu:0')
print(device)
#Set random seed
torch.manual_seed(config_random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config_random_seed)

# Load data
if config_dataset == 'MNIST':
    trainloader, testloader = mnist_loader(batch_size=config_batch_size)
else:
    trainloader, testloader = cifar_loader(batch_size=config_batch_size)

model = MODELS_MAP[config_architecture]()
model = model.to(device)
criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(model.parameters(), lr=config_lr,momentum=config_momentum, weight_decay=config_weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config_epoch)

print_freq = 50
best_prec1 = 0
for epoch in range(0, config_epoch):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(trainloader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(testloader, model, criterion, device)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        
model.to(cpu_dev)
# Deep copy of the model for layer fusion
fused_model = copy.deepcopy(model)

model.eval()
# Swithcing model in evaluation mode before layer fusion
fused_model.eval()
fused_model = torch.quantization.fuse_modules(fused_model, [["First_Conv", "Batch_Norm_1", "Relu"]], inplace=True)
for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["First_Conv", "Batch_Norm_1", "Relu1"], ["Second_Conv", "Batch_Norm_2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
                        
# Inserting qunatization operators on the fused model
quantized_model = ModelQuantization(model=fused_model)
  
# Selecting quantization schemes on the model
backend = config_backend # This backend is used when we want to quantize the model to work on Android devices

quantization_config = torch.quantization.get_default_qconfig(backend)
   
quantized_model.qconfig = quantization_config
    
# Print quantization configurations
print(quantized_model.qconfig)

# Preparing model for calibration
torch.quantization.prepare(quantized_model, inplace=True)

# Calibrating the parameters of the quantization equation
model_calibration(quantized_model, trainloader, cpu_dev)

# We finally qunatize the model
quant_model_final = torch.quantization.convert(quantized_model, inplace=True)

    
# We put the model in evaluation mode
quant_model_final.eval()

# Print quantized model.
print(quant_model_final)

floating_point_prec = validate(testloader, model, criterion, cpu_dev)
integer_point_prec = validate(testloader, quant_model_final, criterion, cpu_dev)

print("The original model accuracy is ", floating_point_prec)
print("Quantized model accuracy is", integer_point_prec)

save_model_and_stats(model,config_architecture,round(measure_inference_latency(model, device = cpu_dev)*1000,2),floating_point_prec,is_quantized=False)
#print(round(measure_inference_latency(model, device = cpu_dev),2))
#print(round(measure_inference_latency(quant_model_final, device = cpu_dev),2))

#print(measure_latency(model,input_shape=(1, 3, 32, 32)))
#print(measure_latency(quant_model_final,input_shape=(1, 3, 32, 32)))
save_model_and_stats(quant_model_final,config_architecture,round(measure_inference_latency(quant_model_final, device = cpu_dev)*1000,2),integer_point_prec,is_quantized=True)


