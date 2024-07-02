# 安装PyTorch

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

# 其他依赖关系

```shell
pip install -r requirement.txt
conda install  torch==0.4.0  scipy==1.01

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

查看命令行参数说明

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
  --gpu GPU             (int, default=0) 指定使用GPU计算，一般0就可以
  --use_cpu             (bool, default=False) Disable CPU training, 设置为True表示使用CPU计算
  --data_path DATA_PATH
                        (str, default=data) 数据文件夹所在位置
  --pmask PMASK         (str, default=pathway_mask.csv) mask矩阵所在位置，这个shape的修改会影响到前面的Pathway_Nodes
  --train TRAIN         (str, default=train.csv) 训练文件位置
  --test TEST           (str, default=test.csv) 测试文件位置
  --valid VALID         (str, default=validation.csv) 验证位置
  --full_data FULL_DATA
                        (str, default=entire_data.csv) 完整数据；用于interp的训练
  --ckpt CKPT           (str, default=) 检查点位置，用于interp的训练
  --grid_search         (bool, default=False)
  --grid_epoch GRID_EPOCH
                        (int, default=3000) Grid Search 时的epoch
  --epoch EPOCH         (int, default=20000) for training
  --debug DEBUG         (default=False)
  --interp_epoch INTERP_EPOCH
                        (int, default=15000) Interpret 训练的最大epoch
  --interp_lr INTERP_LR
                        (float, default=0.03) Interpret 训练的学习率
  --interp_l2 INTERP_L2
                        (float, default=0.001) Interpret 训练的L2
  --do_interp           (bool, default=False) 是否进行# Interpret 训练
  -h, --help            show this help message and exit
```
