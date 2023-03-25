## Post Training Quantization Experiments

There are various different ResNet models trained on a Tesla GPU. The repo contains the details of experiments on quantizing Resent Models. And finally a stats.csv file is created to generate the model files and save the stats of model like latency, accuracy and file size. 


# Usage
More models can be added in the models folder. Alongwith the data in the  dataloader in loader.py. 

`python train.py --config config.json`