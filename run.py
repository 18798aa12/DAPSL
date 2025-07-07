import numpy as np
import torch
import pickle
from problem import get

from lhs import lhs
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform

from rf_optimization_problem import RandomForestMaximizationProblem
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
# -----------------------------------------------------------------------------
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

hv_list = {}
import pandas as pd

# dic = {'zdt1':[0.9994, 6.0576],
#         'zdt2':[0.9994, 6.8960],
#         'zdt3':[0.9994, 6.0571],
#         'dtlz2':[2.8390, 2.9011, 2.8575],
#         'dtlz3':[2421.6427, 1905.2767, 2532.9691],
#         'dtlz4':[3.2675, 2.6443, 2.4263],
#         'dtlz5': [2.6672, 2.8009, 2.8575],
#         'dtlz6':[16.8258, 16.9194, 17.7646],
#         'dtlz7':[0.9984, 0.9961, 22.8114],
#         're1':[2.76322289e+03, 3.68876972e-02],
#         're2':[ 528107.18990952, 1279320.81067113],
#        're3':[ 7.68527849,  7.28609807, 21.50103909],
#        're4':[ 6.79211111, 60.     ,     0.4799612 ],
#        're5':[0.87449713, 1.05091656, 1.05328528],
#        're6':[749.92405125, 2229.37483405],
#        're7':[2.10336300e+02 ,1.06991599e+03, 3.91967702e+07],
#        }

# for test_ins in ins_list:
#     print(test_ins)
    
#     # get problem info
#     hv_all_value = np.zeros([n_run, n_iter+1])
#     if test_ins.startswith('zdt'):
#         problem = get_problem(test_ins, n_var=20)
#     elif test_ins.startswith('dtlz'):
#         problem = get_problem(test_ins, n_var=20, n_obj=3)
#     else:
#         problem = get(test_ins)
#     n_dim = problem.n_var
#     n_obj = problem.n_obj
#     lbound = torch.zeros(n_dim).float()
#     ubound = torch.ones(n_dim).float()
#     ref_point = dic[test_ins]

#     # repeatedly run the algorithm n_run times
#     for run_iter in range(n_run):
#         i_iter = 1
#         x_init = lhs(n_dim, n_init)
#         y_init = problem.evaluate(x_init)
#         p_rel_map, s_rel_map = init_dom_rel_map(300)

#         p_model = init_dom_nn_classifier(x_init, y_init, p_rel_map, pareto_dominance, problem)  # init Pareto-Net
#         evaluated = len(y_init)

#         X = x_init
#         Y = y_init

#         net = GAN(n_dim, 30, 0.0001, 200, n_dim)
#         z = torch.zeros(n_obj).to(device)

#         while evaluated < 200:
#             transformation = StandardTransform([0, 1])
#             transformation.fit(X, Y)
#             X_norm, Y_norm = transformation.do(X, Y)
#             _, index = environment_selection(Y, len(X)//3)
#             real = X[index, :]
#             label = np.zeros((len(Y), 1))
#             label[index, :] = 1
#             net.train(X, label, real)
#             surrogate_model = GaussianProcess(n_dim, n_obj, nu=5)
#             surrogate_model.fit(X_norm, Y_norm)

#             nds = NonDominatedSorting()
#             idx_nds = nds.do(Y_norm)

#             Y_nds = Y_norm[idx_nds[0]]

#             X_gan = net.generate(real / np.tile(ubound, (np.shape(real)[0], 1)), n_init*10) * \
#                           np.tile(ubound, (n_init*10, 1))
#             X_gan = pm_mutation(X_gan, [lbound, ubound])
#             X_ga = random_genetic_variation(real, 1000,list(np.zeros(n_dim)),list(np.ones(n_dim)),n_dim)
#             X_gan = np.concatenate((X_ga, X_gan), axis=0)

#             p_dom_labels, p_cfs = nn_predict_dom_inter(X_gan, real, p_model, device)

            
#             res = np.sum(p_dom_labels,axis=1)
#             iindex = np.argpartition(-res, 100)
#             result_args = iindex[:100]
#             X_dp = X_gan[result_args,:]

#             X_psl = RMMEDA_operator(np.concatenate((X_dp, real), axis=0), 5, n_obj, lbound, ubound)


#             Y_candidate_mean = surrogate_model.evaluate(X_psl)['F']


#             Y_candidata_std = surrogate_model.evaluate(X_psl, std=True)['S']
#             Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
#             Y_candidate_mean = Y_candidate
            
            
            
#             best_subset_list = []
#             Y_p = Y_nds
#             for b in range(n_sample):
#                 hv = HV(ref_point=np.max(np.vstack([Y_p, Y_candidate_mean]), axis=0))
#                 best_hv_value = 0
#                 best_subset = None

#                 for k in range(len(Y_candidate_mean)):
#                     Y_subset = Y_candidate_mean[k]
#                     Y_comb = np.vstack([Y_p, Y_subset])
#                     hv_value_subset = hv(Y_comb)
#                     if hv_value_subset > best_hv_value:
#                         best_hv_value = hv_value_subset
#                         best_subset = [k]

#                 Y_p = np.vstack([Y_p, Y_candidate_mean[best_subset]])
#                 best_subset_list.append(best_subset)
#             best_subset_list = np.array(best_subset_list).T[0]

#             X_candidate = X_psl
#             X_new = X_candidate[best_subset_list]

#             Y_new = problem.evaluate(X_new)
#             Y_new = torch.tensor(Y_new).to(device)

#             X_new = torch.tensor(X_new).to(device)
#             X = np.vstack([X, X_new.detach().cpu().numpy()])
#             Y = np.vstack([Y, Y_new.detach().cpu().numpy()])

#             update_dom_nn_classifier(p_model, X, Y, p_rel_map, pareto_dominance, problem)



#             hv = HV(ref_point=np.array(ref_point))
#             hv_value = hv(Y)
#             hv_all_value[run_iter, i_iter] = hv_value
#             i_iter = i_iter+1
#             print("hv", "{:.4e}".format(np.mean(hv_value)))
#             print("***")
#             evaluated = evaluated + n_sample

#         hv_list[test_ins] = hv_all_value
#         print("************************************************************")

# 只测试随机森林问题
ins_list = ['rf_max']
dic = {
    'rf_max': [-5.0, -5.0]  # 2目标随机森林最大化参考点
}

for test_ins in ins_list:
    print(test_ins)
    
    # get problem info
    hv_all_value = np.zeros([n_run, n_iter+1])
    
    # 直接创建随机森林最大化问题
    problem = RandomForestMaximizationProblem(n_var=10, n_obj=2)
    
    n_dim = problem.n_var
    n_obj = problem.n_obj
    lbound = torch.zeros(n_dim).float()
    ubound = torch.ones(n_dim).float()
    ref_point = dic[test_ins]
    
    # repeatedly run the algorithm n_run times
    for run_iter in range(n_run):
        i_iter = 1
        x_init = lhs(n_dim, n_init)
        y_init = problem.evaluate(x_init)
        p_rel_map, s_rel_map = init_dom_rel_map(300)

        p_model = init_dom_nn_classifier(x_init, y_init, p_rel_map, pareto_dominance, problem)  # init Pareto-Net
        evaluated = len(y_init)

        X = x_init
        Y = y_init

        net = GAN(n_dim, 30, 0.0001, 200, n_dim)
        z = torch.zeros(n_obj).to(device)

        while evaluated < 200:
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
            i_iter = i_iter+1
            print("hv", "{:.4e}".format(np.mean(hv_value)))
            print("***")
            evaluated = evaluated + n_sample

        hv_list[test_ins] = hv_all_value
        print("************************************************************")







