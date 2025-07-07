import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

class RandomForestMaximizationProblem:
    """随机森林最大化问题 - 模仿problem.py中RE类的结构"""
    
    def __init__(self, n_var=10, n_obj=2, n_trees=100):
        self.n_var = n_var
        self.n_obj = n_obj
        self.lbound = torch.zeros(n_var).float()
        self.ubound = torch.ones(n_var).float()
        self.nadir_point = [-5.0] * n_obj  # 用于超体积计算
        
        # 创建训练数据
        X_train, y_train = make_regression(
            n_samples=1000, 
            n_features=n_var, 
            n_targets=n_obj if n_obj > 1 else 1,
            noise=0.1,
            random_state=42
        )
        
        # 归一化训练数据到[0,1]范围
        X_train_min = X_train.min(axis=0)
        X_train_max = X_train.max(axis=0)
        X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
        
        # 训练随机森林模型
        self.rf_models = []
        for i in range(n_obj):
            rf = RandomForestRegressor(
                n_estimators=n_trees,
                max_depth=10,
                random_state=42 + i  # 每个目标使用不同的随机种子
            )
            if n_obj == 1:
                rf.fit(X_train, y_train)
            else:
                rf.fit(X_train, y_train[:, i])
            self.rf_models.append(rf)
    
    def evaluate(self, x):
        """评估函数 - 模仿problem.py中RE类的evaluate方法"""
        x = torch.from_numpy(x).to('cuda')
        
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        # 将输入缩放到[0,1]范围（因为我们的RF模型是在这个范围上训练的）
        x = x * (self.ubound - self.lbound) + self.lbound
        
        # 转换为numpy进行RF预测
        x_np = x.cpu().numpy()
        
        # 用随机森林模型进行预测
        predictions = np.zeros((len(x_np), self.n_obj))
        
        for i, rf_model in enumerate(self.rf_models):
            pred = rf_model.predict(x_np)
            # 转换为最小化问题（因为算法默认是最小化，我们要最大化RF输出）
            predictions[:, i] = -pred  # 负号表示最大化
        
        # 转换回torch tensor然后返回numpy（模仿RE类的模式）
        objs = torch.from_numpy(predictions).float()
        if x.device.type == 'cuda':
            objs = objs.cuda()
        
        return objs.cpu().numpy()

# 添加到problem.py的get函数中
def get_rf_problem(name, *args, **kwargs):
    """获取随机森林优化问题"""
    if name == 'rf_max_2d':
        return RandomForestMaximizationProblem(n_var=10, n_obj=2)
    elif name == 'rf_max_3d':
        return RandomForestMaximizationProblem(n_var=10, n_obj=3)
    else:
        raise Exception("RF Problem not found.")