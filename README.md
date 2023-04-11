## Post Training Quantization Experiments

There are various different ResNet models trained on a Tesla GPU. The repo contains the details of experiments on quantizing Resent Models. And finally a stats.csv file is created to generate the model files and save the stats of model like latency, accuracy and file size. 

## Structure of the repo

models folder contains the ResNet architectures. loader.py has the data loader for CIFAR10 dataset. More dataloaders can be added here. quant.py contains some essential helper functions for inference and saving of models. misc.py has some miscellenius functions. train.py is the main training file. Parameters can be changed in the config.json file. 

## Usage
More models can be added in the models folder. Alongwith the data in the  dataloader in loader.py. 

`python train.py --config config.json`