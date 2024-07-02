# ME-NET

# Install PyTorch

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

# Others 

```shell
pip install -r requirement.txt
conda install  torch  scipy

```

# Run code

```shell
# RUN 'Diagnosis1'
python main.py
# RUN 'Diagnosis2'
python main.py --task Diagnosis2
# RUN Interpret Training
python main.py --do_interp --ckpt example.pt
```

View command line parameter descriptions

```shell
options:
  --In_Nodes IN_NODES   (int, default=1799) number of genes
  --Pathway_Nodes PATHWAY_NODES
                        (int, default=860) number of pathways
  --Hidden_Nodes HIDDEN_NODES
                        (int, default=100) number of hidden nodes
  --Out_Nodes OUT_NODES
                        (int, default=30) number of hidden nodes in the last hidden layer
  --task TASK           (default=Diagnosis1) 'Diagnosis1' or 'Diagnosis2'
  --gpu GPU             (int, default=0) 
  --use_cpu             (bool, default=False) 
  --data_path DATA_PATH
                        (str, default=data) 
  --pmask PMASK         (str, default=pathway_mask.csv) 
  --train TRAIN         (str, default=train.csv) 
  --test TEST           (str, default=test.csv) 
  --valid VALID         (str, default=validation.csv) 
  --full_data FULL_DATA
                        (str, default=entire_data.csv) 
  --ckpt CKPT           (str, default=) 
  --grid_search         (bool, default=False)
  --grid_epoch GRID_EPOCH
                        (int, default=3000) Grid Search 时的epoch
  --epoch EPOCH         (int, default=20000) for training
  --debug DEBUG         (default=False)
  --interp_epoch INTERP_EPOCH
                        (int, default=15000) Interpret 
  --interp_lr INTERP_LR
                        (float, default=0.03) Interpret 
  --interp_l2 INTERP_L2
                        (float, default=0.001) Interpret 
  --do_interp           (bool, default=False) 
  -h, --help            show this help message and exit
```
