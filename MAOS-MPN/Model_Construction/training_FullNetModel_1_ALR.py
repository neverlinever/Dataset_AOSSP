# Description:  仅在1000task的数据集上训练
# Programming Date: 20240527
# Modification Date:
# Programmer: LZ


import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import os
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from NetModel_FullAdvanced_1 import MultiAgentModel, TrainDataset, trend_mean, Adaptive_CosAnneal_WarmRestart

if __name__ == '__main__':

    # 0 加载数据
    num_sat = 10
    num_task = 1000
    path_load = 'F:\PythonTestNew\Data_timewindows\MAOS_Standard_Data_202405\d4_formal_TrainingData' \
                '\DatasetFormal_{}sat_{}task_1920.pt'.format(num_sat, num_task)
    train_data = torch.load(path_load)
    train_data = TrainDataset(train_data)

    # 1 明确当前数据的保存路径
    time_str = '%s' % datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')
    now = time_str.replace('.', '')
    now = now.replace(' ', '_')
    now = now.replace(':', '')
    path_parent = os.path.abspath(os.path.dirname(os.getcwd()))
    path = path_parent + '\TrainingResults\\FullNetmodel_ALR_' + now
    os.mkdir(path)
    fp = open(path_parent + '\TrainingResults\\Training Log.txt', 'a', encoding='utf-8')
    fp.write('\n' + time_str + ' FullNetmodel_ALR 训练结果保存路径：' + path)
    fp.close()
    del time_str
    print('训练结果保存路径：', path)
    path_writer = path + '\\log'
    writer = SummaryWriter(path_writer)

    # 2 参数设置
    # 2.1 卫星参数
    num_sat = 10
    cycle_times = [14.76500445, 14.76600451, 14.76558539, 14.76556004, 14.80824437,
                   14.58938877, 14.57801690, 14.57730479, 14.76530287, 15.21324639,
                   15.24065849, 14.79343693, 15.23303633, 15.23716195, 15.23580491,
                   15.28263578, 15.23639685, 15.23690753, 15.23322993, 14.76764294]
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

    # 2.2 网络模型参数 dim_require, dim_vtw, dim_hidden, num_heads, dp, num_agent
    model_parser = argparse.ArgumentParser(description='Parameters of NetModel')
    model_parser.add_argument('--dim_require', default=2, type=int, help='需求数据维度')
    model_parser.add_argument('--dim_vtw', default=5, type=int, help='窗口数据维度')  # 卫星，轨道，est,lst,ra
    model_parser.add_argument('--dim_hidden', default=64, type=int, help='隐藏层维度')
    model_parser.add_argument('--num_heads', default=4, type=int, help='多头数')
    model_parser.add_argument('--drop_out', default=0.1, type=float, help='神经元停止概率')
    model_parser.add_argument('--num_agent', default=10, type=int, help='最大卫星数对应最多agent数')
    ModelArgs = model_parser.parse_args()

    # 2.3 训练参数
    train_parser = argparse.ArgumentParser(description='Parameters of training')
    train_parser.add_argument('--batch_size', default=16, type=int, help='批大小')
    train_parser.add_argument('--epochs', default=10, type=int, help='完整训练的轮次')
    train_parser.add_argument('--actor_lr', default=0.00005, type=float, help='主网络学习率')
    train_parser.add_argument('--max_grad_norm', default=5., type=float, help='网络参数梯度的范数上限')
    TrainArgs = train_parser.parse_args()

    # 3 构建网络以及训练算法-测试
    actor = MultiAgentModel(ModelArgs.dim_require, ModelArgs.dim_vtw, ModelArgs.dim_hidden, ModelArgs.num_heads,
                            ModelArgs.drop_out, ModelArgs.num_agent)
    actor_optim = optim.Adam(actor.parameters(), lr=TrainArgs.actor_lr)
    step_sum = TrainArgs.epochs * 1920 / TrainArgs.batch_size
    step_size = 10
    actor_scheduler = Adaptive_CosAnneal_WarmRestart(actor_optim, total_steps=step_sum, step_size=step_size,
                                                     deacy_rate=0.001, min_lr=0.000005, min_lr_second=0.00001)
    print('Start training...')
    actor.train()
    actor_losses, actor_rewards, actor_lr_list = [], [], []
    start_time = time.time()
    count_times = 0

    for epoch in range(TrainArgs.epochs):
        start_time0 = time.time()
        list_eval = []
        train_loader = DataLoader(train_data, TrainArgs.batch_size, shuffle=True, drop_last=True)
        for batch_idx, batch in enumerate(train_loader):
            # tensor_require, tensor_wn, tensor_vtw, matrix_orbit, norm_require, norm_global_vtw, norm_local_vtw
            batch_require, batch_nw, batch_vtw, batch_mo, batch_norm_require, batch_norm_gw, batch_norm_lw = batch
            del batch
            # copy_mo = deepcopy(batch_mo.numpy())
            batch_prob, results, result_evaluation, results_delete = actor(batch_require, batch_vtw, batch_nw, batch_mo,
                                                                           batch_norm_require, batch_norm_gw, batch_norm_lw, SatArgs)

            reward = result_evaluation[:, -1]
            logprob = torch.log(batch_prob)
            if (logprob == float('-inf')).any():
                print('出现-inf', torch.where(logprob == float('-inf')))
                logprob[logprob == float('-inf')] = 0
            logprob = logprob.sum(1)  # 避免logprob全是0的情况
            actor_loss = torch.mean((reward - 1).detach() * logprob)
            actor_optim.zero_grad()

            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(actor.parameters(), TrainArgs.max_grad_norm)
            actor_optim.step()

            # 记录数据
            actor_lr = actor_optim.param_groups[0]['lr']
            list_eval.append(result_evaluation.numpy())
            actor_reward = torch.mean(reward).item()
            actor_rewards.append(actor_reward)
            actor_losses.append(actor_loss.item())
            actor_lr_list.append(actor_lr)

            writer.add_scalar('actor\\actor_reward', actor_reward, count_times)
            writer.add_scalar('actor\\actor_loss', actor_loss, count_times)
            writer.add_scalar('actor\\actor_lr', actor_lr, count_times)

            # 更新学习率# actor_scheduler.step()
            if (actor_scheduler.step_count + 1) % step_size == 0:
                trend_actor, mean_actor = trend_mean(actor_rewards[-step_size:])
                actor_scheduler.step(restart=True, param=(trend_actor, mean_actor, 1))
            else:
                actor_scheduler.step(restart=False)
            count_times += 1
        # 每一类数据计算完保存一下
        path_save = path + '\StageResult_sat{}_task{}_epoch{}.pt'.format(num_sat, num_task, epoch)
        torch.save((list_eval, results, results_delete, time.time() - start_time0), path_save)
        path_save = path + '\StageNetModel_sat{}_task{}_epoch{}.pt'.format(num_sat, num_task, epoch)
        torch.save(actor.state_dict(), path_save)
    stop_time = time.time()
    use_time = stop_time-start_time
    writer.close()  # 关闭
    # batch_mo 在网络中发生了改变，外部值也会变化
    # for bi in range(batch_size):
    #     solution_state = check_solution(results[bi], batch_vtw[bi, :, :, :, :], batch_require[bi, :, 0].numpy(),
    #                                     copy_mo[bi, :, :, :, :], SatArgs)
    #     print(bi)
    path_save = path + '\\NetModel_sat{}.pt'.format(num_sat)
    torch.save(actor.state_dict(), path_save)
    path_save = path + '\FinalResult_sat{}.pt'.format(num_sat)
    torch.save((actor_rewards, actor_losses, actor_lr_list, use_time), path_save)
    print('用时：', use_time)

    # 画图
    # 1 actor的损失函数
    x1 = range(len(actor_losses))
    plt.figure(dpi=600, figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(x1, actor_losses, 'o-')
    plt.title('actor_loss')
    # 2 actor的收益值
    x2 = range(len(actor_rewards))
    plt.subplot(3, 1, 2)
    plt.plot(x2, actor_rewards, '.-')
    plt.title('actor_reward')
    # 3 actor学习率
    x3 = range(len(actor_lr_list))
    plt.subplot(3, 1, 3)
    plt.plot(x3, actor_lr_list, '.-')
    plt.title('actor_lr')

    plt_path = path + '\\training_curves.jpg'
    plt.savefig(plt_path)
    plt.show()
