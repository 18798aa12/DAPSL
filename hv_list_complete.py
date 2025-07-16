import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False

class DAPSLResultAnalyzer:
    """DAPSL实验结果分析器"""
    
    def __init__(self, results_base_dir="results"):
        self.results_base_dir = Path(results_base_dir)
        self.experiment_dirs = []
        self.current_experiment = None
        self.find_experiments()
    
    def find_experiments(self):
        """查找所有实验目录"""
        if self.results_base_dir.exists():
            self.experiment_dirs = [d for d in self.results_base_dir.iterdir() 
                                  if d.is_dir() and d.name.startswith('experiment_')]
            self.experiment_dirs.sort(reverse=True)  # 最新的在前
            
            print(f"🔍 找到 {len(self.experiment_dirs)} 个实验:")
            for i, exp_dir in enumerate(self.experiment_dirs):
                print(f"  {i+1}. {exp_dir.name}")
        else:
            print(f"❌ 结果目录不存在: {self.results_base_dir}")
    
    def select_experiment(self, index=0):
        """选择要分析的实验"""
        if 0 <= index < len(self.experiment_dirs):
            self.current_experiment = self.experiment_dirs[index]
            print(f"📂 选择实验: {self.current_experiment.name}")
            return True
        else:
            print(f"❌ 无效的实验索引: {index}")
            return False
    
    def load_experiment_summary(self):
        """加载实验总结"""
        if not self.current_experiment:
            print("❌ 请先选择一个实验")
            return None
        
        summary_file = self.current_experiment / "data" / "experiment_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"⚠️  实验总结文件不存在: {summary_file}")
            return None
    
    def load_hv_data(self):
        """加载HV收敛数据"""
        if not self.current_experiment:
            print("❌ 请先选择一个实验")
            return None
        
        hv_file = self.current_experiment / "data" / "hv_list_complete.pkl"
        if hv_file.exists():
            with open(hv_file, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"⚠️  HV数据文件不存在: {hv_file}")
            return None
    
    def get_latest_population_data(self, problem_name="catboost_min", run_id=0):
        """获取最新的种群数据"""
        if not self.current_experiment:
            print("❌ 请先选择一个实验")
            return None, None
        
        pop_dir = self.current_experiment / "populations" / problem_name
        if not pop_dir.exists():
            print(f"❌ 种群目录不存在: {pop_dir}")
            return None, None
        
        # 查找最新的种群文件
        pop_files = list(pop_dir.glob(f"population_{problem_name}_run{run_id}_iter*.pkl"))
        if not pop_files:
            print(f"❌ 没有找到运行{run_id}的种群文件")
            return None, None
        
        # 按迭代次数排序，取最后一个
        pop_files.sort(key=lambda x: int(x.stem.split('_iter')[1]))
        latest_file = pop_files[-1]
        
        with open(latest_file, 'rb') as f:
            population_data = pickle.load(f)
        
        print(f"📊 加载最新种群数据: {latest_file.name}")
        return population_data, latest_file
    
    def get_latest_pareto_data(self, problem_name="catboost_min", run_id=0):
        """获取最新的Pareto前沿数据"""
        if not self.current_experiment:
            print("❌ 请先选择一个实验")
            return None, None
        
        pareto_dir = self.current_experiment / "pareto_fronts" / problem_name
        if not pareto_dir.exists():
            print(f"❌ Pareto前沿目录不存在: {pareto_dir}")
            return None, None
        
        # 查找最新的Pareto前沿文件
        pareto_files = list(pareto_dir.glob(f"pareto_front_{problem_name}_run{run_id}_iter*.pkl"))
        if not pareto_files:
            print(f"❌ 没有找到运行{run_id}的Pareto前沿文件")
            return None, None
        
        # 按迭代次数排序，取最后一个
        pareto_files.sort(key=lambda x: int(x.stem.split('_iter')[1]))
        latest_file = pareto_files[-1]
        
        with open(latest_file, 'rb') as f:
            pareto_data = pickle.load(f)
        
        print(f"🎯 加载最新Pareto前沿: {latest_file.name}")
        return pareto_data, latest_file
    
    def analyze_final_results(self, problem_name="catboost_min", run_id=0):
        """分析最终结果"""
        print("="*60)
        print(f"📈 分析 {problem_name} 问题的最终结果 (运行 {run_id})")
        print("="*60)
        
        # 加载种群数据
        pop_data, pop_file = self.get_latest_population_data(problem_name, run_id)
        if pop_data is None:
            return None
        
        # 加载Pareto前沿数据
        pareto_data, pareto_file = self.get_latest_pareto_data(problem_name, run_id)
        if pareto_data is None:
            return None
        
        final_X = pop_data['decision_variables']
        final_Y = pop_data['objective_values']
        pareto_front_Y = pareto_data['pareto_front_objectives']
        pareto_indices = pareto_data['pareto_front_indices']
        
        # 获取Pareto最优的参数配置
        pareto_optimal_X = final_X[pareto_indices]
        pareto_optimal_Y = final_Y[pareto_indices]
        
        print(f"\n📊 种群统计信息:")
        print(f"  - 最终种群大小: {len(final_X)}")
        print(f"  - 决策变量维度: {final_X.shape[1]}")
        print(f"  - 目标函数数量: {final_Y.shape[1]}")
        print(f"  - Pareto前沿解数量: {len(pareto_optimal_X)}")
        print(f"  - Pareto前沿占比: {len(pareto_optimal_X)/len(final_X)*100:.2f}%")
        
        # 目标值统计
        print(f"\n🎯 目标值分析:")
        for i in range(final_Y.shape[1]):
            obj_name = f"目标_{i+1}"
            print(f"  {obj_name}:")
            print(f"    全种群范围: [{final_Y[:, i].min():.4f}, {final_Y[:, i].max():.4f}]")
            print(f"    Pareto前沿范围: [{pareto_optimal_Y[:, i].min():.4f}, {pareto_optimal_Y[:, i].max():.4f}]")
            print(f"    Pareto前沿均值: {pareto_optimal_Y[:, i].mean():.4f} ± {pareto_optimal_Y[:, i].std():.4f}")
        
        # 决策变量统计
        print(f"\n🔧 决策变量分析:")
        feature_names = self.get_feature_names()
        for i in range(final_X.shape[1]):
            feat_name = feature_names[i] if i < len(feature_names) else f"变量_{i+1}"
            print(f"  {feat_name}:")
            print(f"    全种群范围: [{final_X[:, i].min():.4f}, {final_X[:, i].max():.4f}]")
            print(f"    Pareto前沿范围: [{pareto_optimal_X[:, i].min():.4f}, {pareto_optimal_X[:, i].max():.4f}]")
            print(f"    Pareto前沿均值: {pareto_optimal_X[:, i].mean():.4f} ± {pareto_optimal_X[:, i].std():.4f}")
        
        return {
            'population_data': pop_data,
            'pareto_data': pareto_data,
            'final_X': final_X,
            'final_Y': final_Y,
            'pareto_optimal_X': pareto_optimal_X,
            'pareto_optimal_Y': pareto_optimal_Y
        }
    
    def get_feature_names(self):
        """获取特征名称"""
        return [
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
            'Livestock NUE'
        ]
    
    def plot_convergence(self, problem_name="catboost_min"):
        """绘制收敛曲线"""
        hv_data = self.load_hv_data()
        if hv_data is None or problem_name not in hv_data:
            print(f"❌ 没有找到 {problem_name} 的HV数据")
            return
        
        hv_matrix = hv_data[problem_name]
        n_runs, n_iters = hv_matrix.shape
        
        plt.figure(figsize=(12, 8))
        
        # 绘制每次运行的曲线
        for i in range(n_runs):
            plt.plot(range(n_iters), hv_matrix[i, :], 
                    alpha=0.7, linewidth=2, marker='o', markersize=4,
                    label=f'Run {i+1}')
        
        # 绘制平均曲线
        mean_hv = np.mean(hv_matrix, axis=0)
        std_hv = np.std(hv_matrix, axis=0)
        plt.plot(range(n_iters), mean_hv, 
                'k-', linewidth=3, marker='s', markersize=6, label='Mean')
        
        # 置信区间
        plt.fill_between(range(n_iters), 
                       mean_hv - std_hv, 
                       mean_hv + std_hv, 
                       alpha=0.2, color='black', label='±1 Std')
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Hypervolume', fontsize=12)
        plt.title(f'{problem_name} - Hypervolume Convergence', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_pareto_front(self, problem_name="catboost_min", run_id=0):
        """绘制Pareto前沿"""
        results = self.analyze_final_results(problem_name, run_id)
        if results is None:
            return
        
        final_Y = results['final_Y']
        pareto_optimal_Y = results['pareto_optimal_Y']
        
        plt.figure(figsize=(10, 8))
        
        # 绘制所有解
        plt.scatter(final_Y[:, 0], final_Y[:, 1], 
                   alpha=0.6, s=30, c='lightblue', label='All Solutions')
        
        # 绘制Pareto前沿
        plt.scatter(pareto_optimal_Y[:, 0], pareto_optimal_Y[:, 1], 
                   alpha=0.8, s=60, c='red', edgecolor='black', 
                   label='Pareto Front', zorder=5)
        
        plt.xlabel('Objective 1', fontsize=12)
        plt.ylabel('Objective 2', fontsize=12)
        plt.title(f'{problem_name} - Pareto Front (Run {run_id})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_parameter_distribution(self, problem_name="catboost_min", run_id=0):
        """绘制参数分布（改进版）"""
        results = self.analyze_final_results(problem_name, run_id)
        if results is None:
            return
        
        final_X = results['final_X']
        pareto_optimal_X = results['pareto_optimal_X']
        feature_names = self.get_feature_names()
        
        n_vars = final_X.shape[1]
        
        # 改进布局：使用3列布局，增加图表高度
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        # 增加图表尺寸和间距
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        fig.suptitle(f'{problem_name} - Parameter Distribution (Run {run_id})', 
                    fontsize=16, y=0.98, fontweight='bold')
        
        # 调整子图间距
        plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.94, bottom=0.06)
        
        # 确保axes是二维数组
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_vars):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # 计算合适的bin数量
            n_bins_all = min(30, max(10, len(final_X) // 10))
            n_bins_pareto = min(20, max(5, len(pareto_optimal_X) // 3))
            
            # 绘制所有解的分布
            ax.hist(final_X[:, i], bins=n_bins_all, alpha=0.6, color='lightblue', 
                   label='All Solutions', density=True, edgecolor='white', linewidth=0.5)
            
            # 绘制Pareto前沿解的分布
            ax.hist(pareto_optimal_X[:, i], bins=n_bins_pareto, alpha=0.8, color='red', 
                   label='Pareto Front', density=True, edgecolor='darkred', linewidth=0.5)
            
            # 改进特征名称显示
            feat_name = feature_names[i] if i < len(feature_names) else f'Variable_{i+1}'
            
            # 处理长标题：换行显示
            if len(feat_name) > 20:
                words = feat_name.split()
                if len(words) > 1:
                    mid = len(words) // 2
                    feat_name = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
            
            ax.set_title(feat_name, fontsize=11, pad=15, fontweight='bold')
            ax.set_xlabel('Normalized Value', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            
            # 改进图例显示
            ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 设置坐标轴范围
            ax.set_xlim(-0.05, 1.05)
            
            # 添加统计信息
            pareto_mean = pareto_optimal_X[:, i].mean()
            ax.axvline(pareto_mean, color='red', linestyle='--', alpha=0.7, linewidth=2)
            
            # 在图上标注均值
            ax.text(0.02, 0.95, f'Pareto Mean: {pareto_mean:.3f}', 
                   transform=ax.transAxes, fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   verticalalignment='top')
        
        # 隐藏多余的子图
        for i in range(n_vars, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        # 保存图像
        if self.current_experiment:
            save_path = self.current_experiment / "plots" / f"parameter_distribution_{problem_name}_run{run_id}_improved.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 参数分布图已保存: {save_path}")
        
        plt.show()
    
    def plot_parameter_distribution_compact(self, problem_name="catboost_min", run_id=0):
        """绘制紧凑版参数分布图"""
        results = self.analyze_final_results(problem_name, run_id)
        if results is None:
            return
        
        final_X = results['final_X']
        pareto_optimal_X = results['pareto_optimal_X']
        feature_names = self.get_feature_names()
        
        n_vars = final_X.shape[1]
        
        # 创建紧凑布局
        fig = plt.figure(figsize=(20, 12))
        
        # 使用GridSpec来更好地控制布局
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle(f'{problem_name} - Parameter Distribution Analysis (Run {run_id})', 
                    fontsize=18, fontweight='bold', y=0.96)
        
        for i in range(min(n_vars, 15)):  # 最多显示15个参数
            row = i // 4
            col = i % 4
            ax = fig.add_subplot(gs[row, col])
            
            # 绘制分布
            ax.hist(final_X[:, i], bins=25, alpha=0.5, color='skyblue', 
                   label='All', density=True, edgecolor='none')
            
            ax.hist(pareto_optimal_X[:, i], bins=15, alpha=0.8, color='crimson', 
                   label='Pareto', density=True, edgecolor='none')
            
            # 简化特征名称
            feat_name = feature_names[i] if i < len(feature_names) else f'Var_{i+1}'
            
            # 提取关键词作为标题
            key_words = {
                'Fertilizer N input intensity': 'Fertilizer N',
                'The proportion of manure fertilizer': 'Manure Ratio',
                'The proportion of BNF in the total N input': 'BNF Ratio',
                'Vegetable and fruit land share': 'Veg&Fruit Share',
                'Crop NUE': 'Crop NUE',
                'Grassland area share': 'Grassland Share',
                'Livestock protein share': 'Livestock Protein',
                'The proportion of the number of layer': 'Layer Ratio',
                'The proportion of the number of meat cattle': 'Meat Cattle',
                'The proportion of the number of meat chicken': 'Meat Chicken',
                'The proportion of the number of dairy': 'Dairy Ratio',
                'The proportion of the number of sheep': 'Sheep Ratio',
                'Livestock NUE': 'Livestock NUE'
            }
            
            short_name = key_words.get(feat_name, feat_name)
            ax.set_title(short_name, fontsize=12, fontweight='bold', pad=10)
            
            # 简化坐标轴
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.tick_params(labelsize=9)
            
            # 添加图例
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3, linestyle=':')
            
            # 添加Pareto均值线
            pareto_mean = pareto_optimal_X[:, i].mean()
            ax.axvline(pareto_mean, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # 添加数值标注
            ax.text(0.05, 0.9, f'μ={pareto_mean:.2f}', 
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # 保存图像
        if self.current_experiment:
            save_path = self.current_experiment / "plots" / f"parameter_distribution_{problem_name}_run{run_id}_compact.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 紧凑版参数分布图已保存: {save_path}")
        
        plt.show()
    
    def plot_parameter_heatmap(self, problem_name="catboost_min", run_id=0):
        """绘制参数热力图"""
        results = self.analyze_final_results(problem_name, run_id)
        if results is None:
            return
        
        final_X = results['final_X']
        pareto_optimal_X = results['pareto_optimal_X']
        feature_names = self.get_feature_names()
        
        # 计算参数统计信息
        stats_data = []
        for i in range(final_X.shape[1]):
            feat_name = feature_names[i] if i < len(feature_names) else f'Var_{i+1}'
            stats_data.append({
                'Parameter': feat_name,
                'Pop_Mean': final_X[:, i].mean(),
                'Pop_Std': final_X[:, i].std(),
                'Pareto_Mean': pareto_optimal_X[:, i].mean(),
                'Pareto_Std': pareto_optimal_X[:, i].std(),
                'Convergence': 1 - pareto_optimal_X[:, i].std(),  # 收敛度指标
            })
        
        df = pd.DataFrame(stats_data)
        
        # 创建热力图
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # 子图1：均值对比
        mean_data = df[['Pop_Mean', 'Pareto_Mean']].T
        mean_data.columns = [f'P{i+1}' for i in range(len(df))]
        
        sns.heatmap(mean_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=axes[0], cbar_kws={'label': 'Mean Value'})
        axes[0].set_title('Parameter Means Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Population Type', fontsize=12)
        
        # 子图2：标准差对比
        std_data = df[['Pop_Std', 'Pareto_Std']].T
        std_data.columns = [f'P{i+1}' for i in range(len(df))]
        
        sns.heatmap(std_data, annot=True, fmt='.3f', cmap='Reds', 
                   ax=axes[1], cbar_kws={'label': 'Standard Deviation'})
        axes[1].set_title('Parameter Std Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Population Type', fontsize=12)
        
        plt.tight_layout()
        
        # 保存图像
        if self.current_experiment:
            save_path = self.current_experiment / "plots" / f"parameter_heatmap_{problem_name}_run{run_id}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 参数热力图已保存: {save_path}")
        
        plt.show()
        
        return df
    
    def generate_comprehensive_report(self, problem_name="catboost_min"):
        """生成综合分析报告（改进版）"""
        print("="*80)
        print(f"📋 生成 {problem_name} 问题的综合分析报告")
        print("="*80)
        
        # 加载实验总结
        summary = self.load_experiment_summary()
        if summary:
            print(f"\n🔬 实验基本信息:")
            exp_info = summary['experiment_info']
            params = summary['parameters']
            print(f"  - 实验时间: {exp_info['start_time']} ~ {exp_info['end_time']}")
            print(f"  - 实验耗时: {exp_info['duration_str']}")
            print(f"  - 运行次数: {params['n_runs']}")
            print(f"  - 迭代次数: {params['n_iterations']}")
            print(f"  - 每次采样: {params['n_samples_per_iter']}")
        
        # 分析每次运行的结果
        all_results = []
        for run_id in range(summary['parameters']['n_runs'] if summary else 2):
            print(f"\n🔄 分析运行 {run_id}:")
            results = self.analyze_final_results(problem_name, run_id)
            if results:
                all_results.append(results)
        
        # 绘制所有可视化图表
        print(f"\n📊 生成可视化图表...")
        self.plot_convergence(problem_name)
        
        for run_id in range(len(all_results)):
            print(f"  - 运行 {run_id} 的图表...")
            self.plot_pareto_front(problem_name, run_id)
            
            # 生成三种不同风格的参数分布图
            print(f"    🎨 生成参数分布图...")
            self.plot_parameter_distribution(problem_name, run_id)  # 改进版
            self.plot_parameter_distribution_compact(problem_name, run_id)  # 紧凑版
            self.plot_parameter_heatmap(problem_name, run_id)  # 热力图版
            
            self.export_results_to_excel(problem_name, run_id)
        
        print(f"\n✅ 综合分析报告生成完成!")
        print(f"📁 所有图表和数据已保存到: {self.current_experiment}")

def main():
    """主函数"""
    print("🎯 DAPSL实验结果分析工具")
    print("="*50)
    
    # 创建分析器
    analyzer = DAPSLResultAnalyzer()
    
    if not analyzer.experiment_dirs:
        print("❌ 没有找到任何实验结果")
        return
    
    # 选择最新的实验
    analyzer.select_experiment(0)
    
    # 生成综合报告
    analyzer.generate_comprehensive_report("catboost_min")

if __name__ == "__main__":
    main()