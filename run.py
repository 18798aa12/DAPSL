import numpy as np
import torch
import pickle
from problem import get
import logging
import sys
from pathlib import Path

from lhs import lhs
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform

from rf_optimization_problem import RandomForestMaximizationProblem
from catboost_optimization_problem import CatBoostMinimizationProblem
from spea2_env import environment_selection
from PM_mutation import pm_mutation
from GAN_model import GAN
from Generate import RMMEDA_operator
from pymop.factory import get_problem
from mating_selection import random_genetic_variation
from evolution.utils import *
from learning.model_init import *
from learning.model_update import *
from learning.prediction import *

# 添加绘图和保存功能
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import json

# -----------------------------------------------------------------------------
# 实验参数设置
mmm = []
ins_list = [
              'zdt1','zdt2','zdt3','dtlz2','dtlz3','dtlz4','dtlz5','dtlz6','dtlz7','re1', 're2','re3','re4','re5','re6','re7'
            ]

# number of independent runs
n_run = 2
# number of initialized solutions
n_init = 100
# number of iterations, and batch size per iteration
n_iter = 20
n_sample = 5 

# PSL 
# number of learning steps
n_steps = 1000 
# number of sampled preferences per step
n_pref_update = 10 
# coefficient of LCB
coef_lcb = 0.1
# number of sampled candidates on the approxiamte Pareto front
n_candidate = 1000 
# number of optional local search
n_local = 1
# device
device = 'cuda'
# -----------------------------------------------------------------------------

# 设置日志和结果保存
def setup_logging_and_directories():
    """设置日志系统和创建结果目录"""
    
    # 创建结果目录结构
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path("results") / f"experiment_{timestamp}"
    
    directories = {
        'base': base_dir,
        'logs': base_dir / "logs",
        'data': base_dir / "data", 
        'plots': base_dir / "plots",
        'pareto_fronts': base_dir / "pareto_fronts",
        'models': base_dir / "models",
        'populations': base_dir / "populations"
    }
    
    # 创建所有目录
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 设置日志配置
    log_file = directories['logs'] / "experiment.log"
    
    # 创建logger
    logger = logging.getLogger('DAPSL_Experiment')
    logger.setLevel(logging.DEBUG)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 文件处理器 - 详细日志
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器 - 简化输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, directories

def save_population_data(X, Y, iteration, run_iter, problem_name, data_dir):
    """保存种群数据"""
    
    # 保存决策变量和目标值
    pop_data = {
        'decision_variables': X,
        'objective_values': Y,
        'iteration': iteration,
        'run': run_iter,
        'population_size': len(X),
        'problem': problem_name,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = data_dir / f"population_{problem_name}_run{run_iter}_iter{iteration}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(pop_data, f)
    
    return filename

def save_pareto_front(Y, iteration, run_iter, problem_name, pareto_dir):
    """提取并保存Pareto前沿"""
    
    # 使用非支配排序提取Pareto前沿
    nds = NonDominatedSorting()
    fronts = nds.do(Y)
    
    pareto_front_indices = fronts[0]
    pareto_front_Y = Y[pareto_front_indices]
    
    # 保存Pareto前沿数据
    pareto_data = {
        'pareto_front_objectives': pareto_front_Y,
        'pareto_front_indices': pareto_front_indices,
        'iteration': iteration,
        'run': run_iter,
        'problem': problem_name,
        'front_size': len(pareto_front_Y),
        'timestamp': datetime.now().isoformat()
    }
    
    filename = pareto_dir / f"pareto_front_{problem_name}_run{run_iter}_iter{iteration}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(pareto_data, f)
    
    # 同时保存为CSV便于查看
    df = pd.DataFrame(pareto_front_Y, columns=[f'Obj_{i+1}' for i in range(pareto_front_Y.shape[1])])
    df['Run'] = run_iter
    df['Iteration'] = iteration
    csv_filename = pareto_dir / f"pareto_front_{problem_name}_run{run_iter}_iter{iteration}.csv"
    df.to_csv(csv_filename, index=False)
    
    return filename, pareto_front_Y

def save_convergence_metrics(hv_all_value, problem_name, data_dir):
    """保存收敛指标"""
    
    metrics = {
        'hypervolume_matrix': hv_all_value,
        'mean_hv': np.mean(hv_all_value, axis=0),
        'std_hv': np.std(hv_all_value, axis=0),
        'best_hv': np.max(hv_all_value, axis=0),
        'worst_hv': np.min(hv_all_value, axis=0),
        'final_improvement': np.mean(hv_all_value[:, -1] - hv_all_value[:, 0]),
        'problem_name': problem_name,
        'n_runs': hv_all_value.shape[0],
        'n_iterations': hv_all_value.shape[1] - 1
    }
    
    filename = data_dir / f"convergence_metrics_{problem_name}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(metrics, f)
    
    return filename

def plot_convergence_with_save(hv_all_value, problem_name, plots_dir, logger):
    """绘制并保存收敛曲线"""
    
    n_run, n_iter_plus1 = hv_all_value.shape
    n_iter = n_iter_plus1 - 1
    
    # 计算统计量
    mean_hv = np.mean(hv_all_value, axis=0)
    std_hv = np.std(hv_all_value, axis=0)
    
    # 绘制收敛曲线
    plt.figure(figsize=(12, 8))
    
    # 绘制每次运行的曲线
    for i in range(n_run):
        plt.plot(range(n_iter + 1), hv_all_value[i, :], 
                alpha=0.6, linewidth=1, label=f'Run {i+1}')
    
    # 绘制平均值曲线
    plt.plot(range(n_iter + 1), mean_hv, 
            'k-', linewidth=3, label='Mean')
    
    # 绘制置信区间
    plt.fill_between(range(n_iter + 1), 
                    mean_hv - std_hv, 
                    mean_hv + std_hv, 
                    alpha=0.2, color='black', label='±1 Std')
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Hypervolume', fontsize=12)
    plt.title(f'{problem_name} - Hypervolume Convergence', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    plot_filename = plots_dir / f"{problem_name}_convergence.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图像以释放内存
    
    logger.info(f"收敛曲线已保存: {plot_filename}")
    return plot_filename

# 设置日志和目录
logger, directories = setup_logging_and_directories()

hv_list = {}

# 只测试CatBoost最小化问题
ins_list = ['catboost_min']
dic = {
    'catboost_min': [0.2, 0.1],
    # 'rf_max': [-5.0, -5.0]  # 最小化问题的参考点 (正值)
}

# 记录实验开始
experiment_start_time = datetime.now()
logger.info("="*80)
logger.info(f"🚀 DAPSL 实验开始")
logger.info(f"开始时间: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"结果保存目录: {directories['base']}")
logger.info("="*80)

# 记录实验参数
logger.info(f"实验参数配置:")
logger.info(f"  - 独立运行次数: {n_run}")
logger.info(f"  - 初始解数量: {n_init}")
logger.info(f"  - 迭代次数: {n_iter}")
logger.info(f"  - 每次迭代采样数: {n_sample}")
logger.info(f"  - LCB系数: {coef_lcb}")
logger.info(f"  - 候选解数量: {n_candidate}")
logger.info(f"  - 计算设备: {device}")

for test_ins in ins_list:
    logger.info("="*60)
    logger.info(f"🔬 开始测试问题: {test_ins}")
    logger.info("="*60)
    
    # get problem info
    hv_all_value = np.zeros([n_run, n_iter+1])
    
    # 创建CatBoost最小化问题
    if test_ins == 'catboost_min':
        problem = CatBoostMinimizationProblem(n_var=13, n_obj=2)
        n_dim = 13
        n_obj = 2
    else:
        problem = RandomForestMaximizationProblem(n_var=10, n_obj=2)
        n_dim = problem.n_var
        n_obj = problem.n_obj
    
    lbound = torch.zeros(n_dim).float()
    ubound = torch.ones(n_dim).float()
    ref_point = dic[test_ins]
    
    logger.info(f"问题信息:")
    logger.info(f"  - 变量维度: {n_dim}")
    logger.info(f"  - 目标数量: {n_obj}")
    logger.info(f"  - 变量边界: [0, 1]^{n_dim}")
    logger.info(f"  - 参考点: {ref_point}")
    logger.info(f"  - 运行次数: {n_run}")
    logger.info(f"  - 迭代次数: {n_iter}")
    
    # 为当前问题创建专门的数据保存目录
    problem_data_dir = directories['data'] / test_ins
    problem_pareto_dir = directories['pareto_fronts'] / test_ins
    problem_pop_dir = directories['populations'] / test_ins
    
    for dir_path in [problem_data_dir, problem_pareto_dir, problem_pop_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # repeatedly run the algorithm n_run times
    for run_iter in range(n_run):
        logger.info(f"\n🔄 开始第 {run_iter+1}/{n_run} 次运行")
        
        i_iter = 1
        x_init = lhs(n_dim, n_init)
        y_init = problem.evaluate(x_init)
        
        # 计算初始HV值
        hv = HV(ref_point=np.array(ref_point))
        initial_hv = hv(y_init)
        hv_all_value[run_iter, 0] = initial_hv
        logger.info(f"  📈 初始HV值: {initial_hv:.4e}")
        
        # 保存初始种群和Pareto前沿
        save_population_data(x_init, y_init, 0, run_iter, test_ins, problem_pop_dir)
        pf_file, pf_data = save_pareto_front(y_init, 0, run_iter, test_ins, problem_pareto_dir)
        logger.debug(f"  初始Pareto前沿大小: {len(pf_data)}")
        
        p_rel_map, s_rel_map = init_dom_rel_map(300)
        p_model = init_dom_nn_classifier(x_init, y_init, p_rel_map, pareto_dominance, problem)
        evaluated = len(y_init)

        X = x_init
        Y = y_init

        net = GAN(n_dim, 30, 0.0001, 200, n_dim)
        z = torch.zeros(n_obj).to(device)

        while evaluated < 200:
            logger.debug(f"    🔍 迭代 {i_iter}, 已评估: {evaluated}")
            
            transformation = StandardTransform([0, 1])
            transformation.fit(X, Y)
            X_norm, Y_norm = transformation.do(X, Y)
            _, index = environment_selection(Y, len(X)//3)
            real = X[index, :]
            label = np.zeros((len(Y), 1))
            label[index, :] = 1
            net.train(X, label, real)
            surrogate_model = GaussianProcess(n_dim, n_obj, nu=5)
            surrogate_model.fit(X_norm, Y_norm)

            nds = NonDominatedSorting()
            idx_nds = nds.do(Y_norm)
            Y_nds = Y_norm[idx_nds[0]]

            X_gan = net.generate(real / np.tile(ubound, (np.shape(real)[0], 1)), n_init*10) * \
                          np.tile(ubound, (n_init*10, 1))
            X_gan = pm_mutation(X_gan, [lbound, ubound])
            X_ga = random_genetic_variation(real, 1000,list(np.zeros(n_dim)),list(np.ones(n_dim)),n_dim)
            X_gan = np.concatenate((X_ga, X_gan), axis=0)

            p_dom_labels, p_cfs = nn_predict_dom_inter(X_gan, real, p_model, device)
            
            res = np.sum(p_dom_labels,axis=1)
            iindex = np.argpartition(-res, 100)
            result_args = iindex[:100]
            X_dp = X_gan[result_args,:]

            X_psl = RMMEDA_operator(np.concatenate((X_dp, real), axis=0), 5, n_obj, lbound, ubound)

            Y_candidate_mean = surrogate_model.evaluate(X_psl)['F']
            Y_candidata_std = surrogate_model.evaluate(X_psl, std=True)['S']
            Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
            Y_candidate_mean = Y_candidate
            
            best_subset_list = []
            Y_p = Y_nds
            for b in range(n_sample):
                hv = HV(ref_point=np.max(np.vstack([Y_p, Y_candidate_mean]), axis=0))
                best_hv_value = 0
                best_subset = None

                for k in range(len(Y_candidate_mean)):
                    Y_subset = Y_candidate_mean[k]
                    Y_comb = np.vstack([Y_p, Y_subset])
                    hv_value_subset = hv(Y_comb)
                    if hv_value_subset > best_hv_value:
                        best_hv_value = hv_value_subset
                        best_subset = [k]

                Y_p = np.vstack([Y_p, Y_candidate_mean[best_subset]])
                best_subset_list.append(best_subset)
            best_subset_list = np.array(best_subset_list).T[0]

            X_candidate = X_psl
            X_new = X_candidate[best_subset_list]

            Y_new = problem.evaluate(X_new)
            Y_new = torch.tensor(Y_new).to(device)

            X_new = torch.tensor(X_new).to(device)
            X = np.vstack([X, X_new.detach().cpu().numpy()])
            Y = np.vstack([Y, Y_new.detach().cpu().numpy()])

            update_dom_nn_classifier(p_model, X, Y, p_rel_map, pareto_dominance, problem)

            hv = HV(ref_point=np.array(ref_point))
            hv_value = hv(Y)
            hv_all_value[run_iter, i_iter] = hv_value
            
            # 计算HV变化
            hv_change = hv_value - hv_all_value[run_iter, i_iter-1] if i_iter > 0 else 0
            logger.debug(f"      📊 HV值: {hv_value:.4e} (变化: {hv_change:+.4e})")
            
            # 保存当前迭代的种群和Pareto前沿
            save_population_data(X, Y, i_iter, run_iter, test_ins, problem_pop_dir)
            pf_file, pf_data = save_pareto_front(Y, i_iter, run_iter, test_ins, problem_pareto_dir)
            logger.debug(f"      当前Pareto前沿大小: {len(pf_data)}")
            
            i_iter = i_iter+1
            evaluated = evaluated + n_sample

        final_hv = hv_all_value[run_iter, -1]
        improvement = final_hv - hv_all_value[run_iter, 0]
        logger.info(f"  ✅ 第 {run_iter+1} 次运行完成")
        logger.info(f"    最终HV值: {final_hv:.4e}")
        logger.info(f"    HV改进: {improvement:+.4e}")
        logger.info(f"    最终种群大小: {len(Y)}")

    # 保存结果到hv_list
    hv_list[test_ins] = hv_all_value
    
    # 计算统计信息
    mean_hv = np.mean(hv_all_value, axis=0)
    std_hv = np.std(hv_all_value, axis=0)
    best_hv = np.max(hv_all_value, axis=0)
    
    logger.info(f"\n📈 {test_ins} 问题结果统计:")
    logger.info(f"  - 最终平均HV: {mean_hv[-1]:.4e} ± {std_hv[-1]:.4e}")
    logger.info(f"  - 最终最佳HV: {best_hv[-1]:.4e}")
    logger.info(f"  - 平均HV改进: {mean_hv[-1] - mean_hv[0]:.4e}")
    logger.info(f"  - 改进百分比: {((mean_hv[-1] - mean_hv[0])/mean_hv[0]*100):.2f}%")
    
    # 保存收敛指标
    metrics_file = save_convergence_metrics(hv_all_value, test_ins, problem_data_dir)
    logger.info(f"  📊 收敛指标已保存: {metrics_file}")
    
    # 保存详细结果到CSV
    results_df = pd.DataFrame(hv_all_value.T, 
                            columns=[f'Run_{i+1}' for i in range(n_run)])
    results_df['Iteration'] = range(n_iter + 1)
    results_df['Mean_HV'] = mean_hv
    results_df['Std_HV'] = std_hv
    results_df['Best_HV'] = best_hv
    
    csv_filename = problem_data_dir / f"{test_ins}_hv_results.csv"
    results_df.to_csv(csv_filename, index=False)
    logger.info(f"  💾 详细HV结果已保存: {csv_filename}")
    
    # 绘制并保存收敛曲线
    plot_file = plot_convergence_with_save(hv_all_value, test_ins, directories['plots'], logger)
    
    logger.info("="*60)

# 保存完整的实验结果
pickle_filename = directories['data'] / "hv_list_complete.pkl"
with open(pickle_filename, 'wb') as f:
    pickle.dump(hv_list, f)

# 保存实验总结
experiment_end_time = datetime.now()
experiment_duration = experiment_end_time - experiment_start_time

summary = {
    'experiment_info': {
        'start_time': experiment_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': experiment_end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_seconds': experiment_duration.total_seconds(),
        'duration_str': str(experiment_duration),
        'base_directory': str(directories['base'])
    },
    'parameters': {
        'n_runs': n_run,
        'n_init': n_init,
        'n_iterations': n_iter,
        'n_samples_per_iter': n_sample,
        'coef_lcb': coef_lcb,
        'n_candidate': n_candidate,
        'device': device
    },
    'problems_tested': list(ins_list),
    'results_summary': {}
}

for problem_name, hv_data in hv_list.items():
    mean_hv = np.mean(hv_data, axis=0)
    std_hv = np.std(hv_data, axis=0)
    summary['results_summary'][problem_name] = {
        'initial_hv_mean': float(mean_hv[0]),
        'initial_hv_std': float(std_hv[0]),
        'final_hv_mean': float(mean_hv[-1]),
        'final_hv_std': float(std_hv[-1]),
        'improvement_mean': float(mean_hv[-1] - mean_hv[0]),
        'improvement_percentage': float((mean_hv[-1] - mean_hv[0])/mean_hv[0]*100),
        'best_hv': float(np.max(hv_data)),
        'worst_hv': float(np.min(hv_data[:, -1])),
        'convergence_stability': float(std_hv[-1]/mean_hv[-1])  # 变异系数
    }

summary_filename = directories['data'] / "experiment_summary.json"
with open(summary_filename, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# 保存文件清单
file_manifest = {
    'directories': {k: str(v) for k, v in directories.items()},
    'key_files': {
        'experiment_log': str(directories['logs'] / "experiment.log"),
        'complete_results': str(pickle_filename),
        'experiment_summary': str(summary_filename),
        'convergence_plots': str(directories['plots']),
        'pareto_fronts': str(directories['pareto_fronts']),
        'population_data': str(directories['populations'])
    },
    'file_counts': {
        'pareto_front_files': len(list(directories['pareto_fronts'].rglob("*.pkl"))),
        'population_files': len(list(directories['populations'].rglob("*.pkl"))),
        'plot_files': len(list(directories['plots'].rglob("*.png")))
    }
}

manifest_filename = directories['base'] / "file_manifest.json"
with open(manifest_filename, 'w', encoding='utf-8') as f:
    json.dump(file_manifest, f, indent=2, ensure_ascii=False)

logger.info("="*80)
logger.info("🏁 实验完成!")
logger.info(f"  ⏱️  总耗时: {experiment_duration}")
logger.info(f"  📁 结果保存目录: {directories['base']}")
logger.info(f"  📋 实验总结: {summary_filename}")
logger.info(f"  💾 完整结果: {pickle_filename}")
logger.info(f"  📊 文件清单: {manifest_filename}")
logger.info(f"  📈 日志文件: {directories['logs'] / 'experiment.log'}")

# 输出文件结构总结
logger.info(f"\n📂 生成的文件结构:")
logger.info(f"  📁 {directories['base']}/")
logger.info(f"    📁 data/ - 实验数据和指标")
logger.info(f"    📁 logs/ - 详细日志文件")
logger.info(f"    📁 plots/ - 收敛曲线图")
logger.info(f"    📁 pareto_fronts/ - Pareto前沿数据")
logger.info(f"    📁 populations/ - 种群演化数据")
logger.info(f"    📁 models/ - 训练的模型(如有)")

logger.info("="*80)

# 最终的简要统计输出到控制台
print(f"\n🎉 实验成功完成!")
print(f"📁 结果保存在: {directories['base']}")
print(f"⏱️  耗时: {experiment_duration}")
for problem_name, summary_data in summary['results_summary'].items():
    print(f"📊 {problem_name}: HV改进 {summary_data['improvement_percentage']:.2f}%")

# 或者交互式使用
from hv_list_complete import DAPSLResultAnalyzer
analyzer = DAPSLResultAnalyzer()
analyzer.select_experiment(0)  # 选择最新实验
analyzer.generate_comprehensive_report("catboost_min")





