from tap import Tap

class Params(Tap):
    path_0: str = 'CD/'  #每次这个都要改“'AD', 'ALS', 'BP', 'CD', 'COPD', 'MS', 'PD', 'RA', 'SCZ'”
    #记得保存
    input: int = 1799  # number of genes
    Pathway_Nodes: int = 2659  # number of pathways
    Hidden_layer: int = 100  # number of hidden nodes
    output: int = 30  # number of hidden nodes in the last hidden layer
    task = 'Diagnosis1'  # 'Diagnosis1' or 'Diagnosis2'
    gpu: int = 0  # 指定使用GPU计算，一般0就可以
    use_cpu: bool = False  # Disable CPU training, 设置为True表示使用CPU计算
    data_path: str = 'data'  # 数据文件夹所在位置
    pmask: str = 'pathway_mask_long1.csv'  # mask矩阵所在位置，这个shape的修改会影响到前面的Pathway_Nodes
    train: str = path_0+'entire_data.csv'  # 训练文件位置
    test: str  = path_0+'TPM_YTlog_test.csv'  #TPM_YTlog21-1.csv 测试文件位置;或者是新加入的数据
    valid: str = path_0+'TPM_YTlog_test.csv'  # 验证位置
    full_data: str = path_0+'entire_data.csv'  # 完整数据；用于interp的训练
    ckpt: str = ''  # 检查点位置，用于interp的训练
    grid_search: bool = False
    grid_epoch: int = 100  # Grid Search 时的epoch,默认3000
    epoch: int = 20000  # for training
    debug = False
    interp_epoch: int = 2500#1000  # Interpret 训练的最大epoch
    interp_lr: float = 0.00075  # Interpret 训练的学习率cd-real 0.03
    interp_l2: float = 0.001#0.001  # Interpret 训练的L2cd-real 0.1
    do_interp: bool = False  # 是否进行# Interpret 训练
    do_test: bool = False # 是否进行# Interpret 训练
    do_load: bool = False # 是否进行# load 训练
#测试的时候train的eval_every设置为2


"""
from tap import Tap

class Params(Tap):
    path_0: str = 'CD/'  #每次这个都要改“'AD', 'ALS', 'BP', 'CD', 'COPD', 'MS', 'PD', 'RA', 'SCZ'”
    #记得保存
    In_Nodes: int = 1799  # number of genes
    Pathway_Nodes: int = 2659  # number of pathways
    Hidden_Nodes: int = 100  # number of hidden nodes
    Out_Nodes: int = 30  # number of hidden nodes in the last hidden layer
    task = 'Diagnosis1'  # 'Diagnosis1' or 'Diagnosis2'
    gpu: int = 0  # 指定使用GPU计算，一般0就可以
    use_cpu: bool = False  # Disable CPU training, 设置为True表示使用CPU计算
    data_path: str = 'data'  # 数据文件夹所在位置
    pmask: str = 'pathway_mask_long1.csv'  # mask矩阵所在位置，这个shape的修改会影响到前面的Pathway_Nodes
    train: str = path_0+'train.csv'  # 训练文件位置
    test: str  = path_0+'TPM_YTlog_test.csv'  #TPM_YTlog21-1.csv 测试文件位置;或者是新加入的数据
    valid: str = path_0+'TPM_YTlog_test.csv'  # 验证位置
    full_data: str = 'entire_data.csv'  # 完整数据；用于interp的训练
    ckpt: str = ''  # 检查点位置，用于interp的训练
    grid_search: bool = False
    grid_epoch: int = 3000  # Grid Search 时的epoch
    epoch: int = 20000  # for training
    debug = False
    interp_epoch: int = 3000#1000  # Interpret 训练的最大epoch
    interp_lr: float = 0.00075  # Interpret 训练的学习率cd-real 0.03
    interp_l2: float = 0.001  # Interpret 训练的L2cd-real 0.1
    do_interp: bool = False  # 是否进行# Interpret 训练
    do_test: bool = False # 是否进行# Interpret 训练
    do_load: bool = False # 是否进行# load 训练
#测试的时候train的eval_every设置为2



"""