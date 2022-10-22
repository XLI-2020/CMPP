
# Modeling and Monitoring of Indoor Populations using Sparse Positioning Data

This is the repository of our work which encompasses supplemental material and implementation details.
# Supplemental Material
The results of supplemental experiments are available in [Supplemental Material](https://anonymous.4open.science/r/CMPP-F823/supplemental_material.pdf), which consist of experimental results of query processing in BLD-2, and \eta's effect in BLD-1. Please download it to local for better readability.

# Implementation Details
The implementation concerns 4 folders which are: 
- preprocess: generate probabilistic populations.
- estimators: population prediction models
- input: population dataset
- query: CMPP query

### Requirements
- Pytorch 1.8.1
- Numpy 1.19.2
- Pandas 1.1.3
- Sklearn 0.24.1
- Matplotlib 3.3.2
- networkx 2.8.4

You may use " pip3 install -r requirements.txt" to install the above libraries.

### Usage
To generate probabilistic populations:
```
cd ./preprocess; nohup python3 -u generate_populations.py > population.log &
```
To train and test population prediction models (e.g., ME):
``` 
cd ../estimators/; nohup python  -u ME.py --epochs 500 --batch_size 64 --time_interval five_mins > ME.log  &
```
To run CMPP query (e.g., varying \theta):
``` 
cd ../query; java -cp CMPP-0.0.1-SNAPSHOT.jar experiments.Experiment_Pop
```
or you may run it  in IDE (e.g., IntelliJ IDEA). Multiple csv files will be generated to record the f1-score, response time, and memory usage respectively.

### Explaination of Parameters

time_interval: prediction time interval (i.e., \delta)

batch_size: the number of samples for back propagation in one pass

epochs: the number of training rounds

### Datasets
The whole datasets are very large and thus moved to [Google Drive](https://drive.google.com/drive/folders/1Vzhg8hQMSdNxQs2AEcbhH3952KaoYHbt?usp=sharing). 

### Acknowledgements
We appreciate [ASTGNN](https://github.com/guoshnBJTU/ASTGNN) and [STGCN](https://github.com/FelixOpolka/STGCN-PyTorch) for publishing codes for ASTGNN and STGCN models respectively. They serve as 
baselines after being adapted into our application scenario.



