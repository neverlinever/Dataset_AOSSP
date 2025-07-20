# Description:  在大数据集上测试训练后的网络
# Programming Date: 20240604
# Modification Date:20240626在10颗星，对比网络做消融实验
# Modification Date:20240720在10颗星，在5个样本下的对比网络做消融实验
# Programmer: LZ

import numpy as np
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import argparse
import os
import time
import datetime
import matplotlib.pyplot as plt

from NetModel_FullAdvanced_1 import TrainDataset
from NetModel_Comparison_1 import MultiAgentModel_noLocalFeature, MultiAgentModel_noPointerSort, MultiAgentModel_Greedy


if __name__ == '__main__':

    # 2 参数设置
    # 2.1 卫星参数
    cycle_times = [14.76500445, 14.76600451, 14.76558539, 14.76556004, 14.80824437,
                   14.58938877, 14.57801690, 14.57730479, 14.76530287, 15.21324639,
                   15.24065849, 14.79343693, 15.23303633, 15.23716195, 15.23580491,
                   15.28263578, 15.23639685, 15.23690753, 15.23322993, 14.76764294]
    # 2.2 网络模型参数 dim_require, dim_vtw, dim_hidden, num_heads, dp, num_agent
    model_parser = argparse.ArgumentParser(description='Parameters of NetModel')
    model_parser.add_argument('--dim_require', default=2, type=int, help='需求数据维度')
    model_parser.add_argument('--dim_vtw', default=5, type=int, help='窗口数据维度')  # 卫星，轨道，est,lst,ra
    model_parser.add_argument('--dim_hidden', default=64, type=int, help='隐藏层维度')
    model_parser.add_argument('--num_heads', default=4, type=int, help='多头数')
    model_parser.add_argument('--drop_out', default=0.1, type=float, help='神经元停止概率')
    model_parser.add_argument('--num_agent', default=10, type=int, help='最大卫星数对应最多agent数')
    ModelArgs = model_parser.parse_args()

    # 3 构建网络以及训练算法-测试
    # MultiAgentModel_noLocalFeature, MultiAgentModel_noPointerSort, MultiAgentModel_Greedy
    # algorithm_name = 'MPN_noPointerSort'
    # actor = MultiAgentModel_noPointerSort(ModelArgs.dim_require, ModelArgs.dim_vtw, ModelArgs.dim_hidden, ModelArgs.num_heads,
    #                         ModelArgs.drop_out, ModelArgs.num_agent)
    # 'MPN_noLocalFeature':
    # dim_require, dim_vtw, dim_hidden, num_heads, dp = 0.1, num_agent = 20
    # 输入： dr, dw, dn, mo, nr, ngw, sat_arg
    # algorithm_name = 'MPN_noLocalFeature'
    # actor = MultiAgentModel_noLocalFeature(ModelArgs.dim_require, ModelArgs.dim_vtw, ModelArgs.dim_hidden, ModelArgs.num_heads,
    #                         ModelArgs.drop_out, ModelArgs.num_agent)

    # 'MPN_Greedy':
    #     dim_require, dim_vtw, dim_hidden, num_heads, dp=0.1, num_agent=20
    # 输入 dr, dw, dn, mo, nr, ngw, nlw, sat_arg
    algorithm_name = 'MPN_Greedy'
    actor = MultiAgentModel_Greedy(ModelArgs.dim_require, ModelArgs.dim_vtw, ModelArgs.dim_hidden,
                                           ModelArgs.num_heads, ModelArgs.drop_out, ModelArgs.num_agent)
    # 加载网络模型参数
    path_parent = os.path.abspath(os.path.dirname(os.getcwd()))
    actor_param = torch.load(path_parent + '\\NetModel_save\{}_sat10_0.pt'.format(algorithm_name))

    actor.load_state_dict(actor_param)
    actor.eval()
    batch_size = 1

    # 0 加载数据
    num_sample = 5
    num_sat = 10
    sat_parser = argparse.ArgumentParser(description='Parameters of AOSs')
    sat_parser.add_argument('--sat_list', default=list(range(num_sat)), type=list, help='卫星编号列表')
    sat_parser.add_argument('--orbit_times', default=cycle_times[:num_sat], type=list, help='24h内轨道圈次')
    sat_parser.add_argument('--orbit_period', default=[24 * 60 * 60 / i for i in cycle_times[:num_sat]], type=list,
                            help='平均轨道周期')
    del cycle_times
    sat_parser.add_argument('--energy', default=1500, type=float)
    sat_parser.add_argument('--memory', default=1000, type=float)
    sat_parser.add_argument('--eco_rate', default=1, type=float)  # 观测时能量消耗速率 *时间
    sat_parser.add_argument('--ect_rate', default=0.5, type=float)  # 姿态转换时能量消耗速率 *度数
    sat_parser.add_argument('--mc_rate', default=1, type=float)  # 内存消耗速率    *时间
    sat_parser.add_argument('--max_pitch', default=45, type=float)  # 俯仰角
    sat_parser.add_argument('--min_pitch', default=-45, type=float)  # 俯仰角
    sat_parser.add_argument('--max_roll', default=45, type=float)  # 滚动角
    sat_parser.add_argument('--min_roll', default=-45, type=float)  # 滚动角
    sat_parser.add_argument('--AngleVelocity', default=3, type=float)  # 最大角速度
    sat_parser.add_argument('--AngleAcceleration', default=1, type=float)  # 角加速度
    sat_parser.add_argument('--period', default=60 * 60 * 24, type=float)  # 调度周期
    SatArgs = sat_parser.parse_args()

    list_task_num = [600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    for num_task in list_task_num:
        path_load = 'F:\PythonTestNew\Data_timewindows\MAOS_Standard_Data_202405\\test1_FinalData' \
                    '\DatasetTesting_{}sat_{}task_{}.pt'.format(num_sat, num_task, num_sample)

        # a_require, a_nw, a_vtw, a_mo, a_norm_require, a_norm_gw, a_norm_lw, _ \
        #     = torch.load(path_load)
        a_data = torch.load(path_load)
        test_data = TrainDataset(a_data)

        # 1 明确当前数据的保存路径
        time_str = '%s' % datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')
        now = time_str.replace('.', '')
        now = now.replace(' ', '_')
        now = now.replace(':', '')

        path = path_parent + '\TestingResults\\MPN_ComparisonNetmodel_0'
        # os.mkdir(path)
        fp = open(path_parent + '\TestingResults\\Testing Log.txt', 'a', encoding='utf-8')
        fp.write('\n' + time_str + ' Comparison_{}_{}sat_{}task_sample 测试结果保存路径：{}'.
                 format(algorithm_name, num_sat, num_task, num_sample))
        fp.close()
        del time_str
        print(num_sat, num_task, num_sample, ' 测试结果保存路径：', path)
        test_results = []
        fitness = np.zeros(num_sample)
        test_loader = DataLoader(test_data, batch_size, shuffle=False, drop_last=True)
        print('Start testing...')
        start_time = time.time()
        for batch_idx, batch in enumerate(test_loader):
            # tensor_require, tensor_wn, tensor_vtw, matrix_orbit, norm_require, norm_global_vtw, norm_local_vtw

            batch_require, batch_nw, batch_vtw, batch_mo, batch_norm_require, batch_norm_gw, batch_norm_lw = batch

            del batch
            start_time0 = time.time()
            # noPointerSort 和Greedy：
            _, results, result_evaluation, results_delete = actor(batch_require, batch_vtw, batch_nw, batch_mo,
                                                                  batch_norm_require, batch_norm_gw, batch_norm_lw, SatArgs)
            # noLocalFeature: dr, dw, dn, mo, nr, ngw, sat_arg
            # _, results, result_evaluation, results_delete = actor(batch_require, batch_vtw, batch_nw, batch_mo,
            #                                                                batch_norm_require, batch_norm_gw, SatArgs)
            use_time = time.time() - start_time0
            test_results.append([results, result_evaluation, results_delete, use_time])
            fitness[batch_idx*batch_size: (batch_idx+1)*batch_size] = deepcopy(result_evaluation[:, -1])
            # print('Complete:', batch_idx, use_time, result_evaluation[:, -1].mean())
        # 至此所有样本测试完成
        use_time = time.time() - start_time
        path_save = path + '\{}_sat{}_task{}_sample{}.pt'.\
            format(algorithm_name, num_sat, num_task, num_sample)
        torch.save((test_results, fitness, use_time), path_save)
        print('Complete:', path_save, use_time, fitness.mean())

