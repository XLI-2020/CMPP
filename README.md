# CMPP
This is the implementation for CMPP project. The query folder contains the code for CMPP query, 
while the others are used for implementing population prediction models

# Requirements

- Pytorch 1.8.1
- Numpy 1.19.2
- Pandas 1.1.3
- Sklearn 0.24.1

#To run the code
To train and evaluate models (e.g., ME):
``` 
nohup python  -u ME.py >> ME.log 2>&1 &
```
To run others models, just replace ME.py with corresponding .py files. For example,
``` 
cd ./baselines
nohup python  -u tgcn.py >> tgcn.log 2>&1 &
```

To run CMPP query experiments (e.g., varying \theta):
``` 
cd ./query
java -cp CMPP-0.0.1-SNAPSHOT.jar experiments.Experiment_Pop
```
or you can open the query folder in IDE (e.g., IntelliJ IDEA) and run them in IDE.

#Notes
- We only provide a sample data set here as the size of the whole data sets (> 3GB) far exceeds the allowed maximum size (i.e., 100MB). We will try to figure out
other ways to publicize the whole data sets.
- Any question please contact xiaol@ruc.dk 
#Acknowledgements
We appreciate [Shengnan Guo](https://github.com/guoshnBJTU/ASTGNN) and [Felix Opolka](https://github.com/FelixOpolka/STGCN-PyTorch) for publishing codes for ASTGNN and STGCN models respectively. They serve as 
baselines after being adapted into our application scenario.



