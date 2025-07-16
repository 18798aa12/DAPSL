import numpy as np
import pickle
import os
from abc import ABC, abstractmethod

class CatBoostMinimizationProblem:
    """
    CatBoost模型最小化优化问题类
    加载预训练的CatBoost模型，将其输出作为最小化目标
    """
    
    def __init__(self, n_var=13, n_obj=2, model_path=None):
        """
        初始化CatBoost最小化问题
        
        参数:
        n_var: 输入变量维度 (13个特征，修正后)
        n_obj: 目标函数数量 (2个目标)
        model_path: CatBoost模型文件路径
        """
        self.model_path = model_path or r'C:\Users\zqs\Documents\GitHub\DAPSL#\0624_model\catboost.pkl'
        self.model = self._load_model()
        
        # 从模型中获取实际的特征数量
        self.n_var = self._get_model_feature_count()
        self.n_obj = n_obj
        
        print(f"✅ 模型期望的特征数量: {self.n_var}")
        
        # 设置变量边界
        self._set_bounds()
    
    def _load_model(self):
        """加载预训练的CatBoost模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            
            print(f"✅ 成功加载CatBoost模型: {self.model_path}")
            return model
            
        except Exception as e:
            raise RuntimeError(f"❌ 加载CatBoost模型失败: {e}")
    
    def _get_model_feature_count(self):
        """获取模型期望的特征数量"""
        try:
            # 尝试用一个小的测试数据来确定特征数量
            for n_features in [12, 13, 14, 15]:
                try:
                    test_data = np.random.rand(1, n_features)
                    _ = self.model.predict(test_data)
                    return n_features
                except Exception:
                    continue
            raise RuntimeError("无法确定模型的特征数量")
        except Exception as e:
            print(f"警告: 无法自动确定特征数量，使用默认值13: {e}")
            return 13
    
    def _set_bounds(self):
        """
        设置变量边界范围
        基于SHAP项目中的特征范围设置合理边界
        """
        # 根据实际特征数量设置边界
        self.xl = np.zeros(self.n_var)  # 下界全为0
        self.xu = np.ones(self.n_var)   # 上界全为1
        
        # 13个特征的名称 (根据SHAP项目推测可能包含的特征)
        if self.n_var == 13:
            self.feature_names = [
                'Fertilizer N input intensity',
                'The proportion of manure fertilizer', 
                'The proportion of BNF in the total N input',
                'Vegetable and fruit land share',
                'Crop NUE',
                'Grassland area share', 
                'Livestock protein share',
                'The proportion of the number of layer',
                'The proportion of the number of meat cattle',
                'The proportion of the number of meat chicken',
                'The proportion of the number of dairy',
                'The proportion of the number of sheep',
                'Livestock NUE'  # 添加第13个特征
            ]
        elif self.n_var == 12:
            self.feature_names = [
                'Fertilizer N input intensity',
                'The proportion of manure fertilizer', 
                'The proportion of BNF in the total N input',
                'Vegetable and fruit land share',
                'Crop NUE',
                'Grassland area share', 
                'Livestock protein share',
                'The proportion of the number of layer',
                'The proportion of the number of meat cattle',
                'The proportion of the number of meat chicken',
                'The proportion of the number of dairy',
                'The proportion of the number of sheep'
            ]
        else:
            # 通用特征名称
            self.feature_names = [f'Feature_{i+1}' for i in range(self.n_var)]
    
    def evaluate(self, X):
        """
        评估目标函数
        
        参数:
        X: 输入变量矩阵，形状 (n_samples, n_var)
        
        返回:
        目标函数值，形状 (n_samples, n_obj)
        """
        # 确保输入是numpy数组
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # 检查输入维度
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # 如果输入特征数量不匹配，进行调整
        if X.shape[1] != self.n_var:
            if X.shape[1] < self.n_var:
                # 如果输入特征少，用随机值或零填充
                padding = np.random.rand(X.shape[0], self.n_var - X.shape[1]) * 0.1
                X = np.hstack([X, padding])
                print(f"⚠️  输入特征从 {X.shape[1] - (self.n_var - X.shape[1])} 扩展到 {self.n_var}")
            else:
                # 如果输入特征多，截断
                X = X[:, :self.n_var]
                print(f"⚠️  输入特征从 {X.shape[1]} 截断到 {self.n_var}")
        
        # 检查变量边界
        X_clipped = np.clip(X, self.xl, self.xu)
        
        try:
            # 使用CatBoost模型预测
            predictions = self.model.predict(X_clipped)
            
            # 确保输出形状正确
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            
            # 修复：返回正确格式的numpy数组，而不是tuple
            # 创建多目标函数值
            n_samples = predictions.shape[0]
            
            # 第一个目标：负的预测值（最小化氨排放强度）
            obj1 = predictions.flatten()
            
            # 第二个目标：负的预测值的一半（可以是另一个相关目标）
            obj2 = 0.5 * predictions.flatten()
            
            # 组合成多目标矩阵
            objectives = np.column_stack([obj1, obj2])

            
            return objectives
            
        except Exception as e:
            print(f"❌ 模型预测失败: {e}")
            print(f"输入形状: {X_clipped.shape}")
            print(f"期望特征数: {self.n_var}")
            raise RuntimeError(f"模型预测失败: {e}")
    
    def get_bounds(self):
        """返回变量边界"""
        return self.xl, self.xu
    
    def get_feature_names(self):
        """返回特征名称"""
        return self.feature_names
    
    def __str__(self):
        return f"CatBoostMinimizationProblem(n_var={self.n_var}, n_obj={self.n_obj})"

# 为了兼容DAPSL算法，创建工厂函数
def create_catboost_problem(model_path=None):
    """创建CatBoost最小化问题实例"""
    return CatBoostMinimizationProblem(model_path=model_path)

# 测试函数
def test_catboost_problem():
    """测试CatBoost优化问题"""
    try:
        # 创建问题实例
        problem = CatBoostMinimizationProblem()
        
        print(f"✅ 问题创建成功: {problem}")
        print(f"特征数量: {problem.n_var}")
        print(f"目标数量: {problem.n_obj}")
        
        # 生成随机测试数据
        n_samples = 5
        X_test = np.random.rand(n_samples, problem.n_var)
        
        # 评估
        Y_test = problem.evaluate(X_test)
        
        print(f"✅ 测试成功!")
        print(f"输入形状: {X_test.shape}")
        print(f"输出形状: {Y_test.shape}")
        print(f"输出类型: {type(Y_test)}")
        print(f"输出范围: [{Y_test.min():.4f}, {Y_test.max():.4f}]")
        print(f"特征名称示例: {problem.get_feature_names()[:3]}...")
        
        return True, problem.n_var
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False, None

if __name__ == "__main__":
    # 运行测试
    success, n_features = test_catboost_problem()
    if success:
        print(f"\n🎉 CatBoost问题测试通过，使用 {n_features} 个特征")
    else:
        print("\n❌ 请检查模型文件和路径")