# Description:  网络的主体结构，包括嵌入层、编码器和解码器
# 编写时间：2024-5-20
# 修改时间：20240529 加入局部选择，每次选择单轨任务规划
# 修改时间：20240609 创建对比网络，用于消融实验
# 作者：LZ


import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

from Calculate_Observation_Action import calculate_EarliestObservationAction
# import check_solution

# 1.嵌入层配合编码器使用
class Embedding(nn.Module):
    def __init__(self, dim_require, dim_vtw, dim_hidden):
        super(Embedding, self).__init__()

        self.dim_half = dim_hidden//2
        # self.dim_require = dim_require
        # self.dim_vtw = dim_vtw
        # self.dim_hidden = dim_hidden
        self.conv1d_require = nn.Conv1d(dim_require, self.dim_half, kernel_size=1)  # （batch, channel, seq）
        self.lstm_vtw = nn.LSTM(dim_vtw, self.dim_half, batch_first=True)
        self.conv1d = nn.Conv1d(dim_hidden, dim_hidden, kernel_size=1)
        self.layer_norm = nn.LayerNorm(dim_hidden)
        # self.dropout = nn.Dropout(dp), dp

    def forward(self, xr, xw, xn):
        """

        :param xr:
        :param xw:
        :param xn: bs, sl, ns+1
        :return:
        """

        bs, sl, _ = xn.size()
        br = F.relu(self.conv1d_require(xr.transpose(1, 2)).transpose(1, 2))  # (batch, seq, hidden)
        bw = torch.zeros([bs, sl, self.dim_half])
        for bi in range(bs):
            max_nw = xn[bi, :, -1].max()    # 最大窗口数
            min_nw = xn[bi, :, -1].min()    # 最小窗口数
            for nw in range(min_nw, max_nw + 1):
                t_indexes = torch.where(xn[bi, :, -1] == nw)[0]
                if t_indexes.size(0) > 0:
                    mbw = xw[bi, t_indexes, :nw, :]  # (mini batch, num vtw, feature)
                    _, (h_t, _) = self.lstm_vtw(mbw)  # h_t隐藏层最终输出，(mini batch,seq=nw, hidden)
                    bw[bi, t_indexes, :] = h_t.squeeze()
        bwr = self.conv1d(torch.cat((br, bw), 2).transpose(1, 2)).transpose(1, 2)   # (batch, seq, hidden)
        # # out = self.dropout(self.layer_norm(bwr))  # (batch, seq, hidden)
        out = self.layer_norm(bwr)
        return out  # batch, seq, feature


# 1.1 需求嵌入
class RequireEmbedding(nn.Module):
    def __init__(self, dim_require, dim_hidden):
        super(RequireEmbedding, self).__init__()
        self.dim_require = dim_require
        self.dim_hidden = dim_hidden
        self.conv1d_require = nn.Conv1d(dim_require, dim_hidden, kernel_size=1)  # （batch, channel, seq）

    def forward(self, xr):
        br = F.relu(self.conv1d_require(xr.transpose(1, 2)).transpose(1, 2))  # (batch, seq, hidden)
        return br


# 1.2 全局窗口信息的嵌入
class FullEmbedding(nn.Module):
    def __init__(self, dim_vtw, dim_hidden):
        super(FullEmbedding, self).__init__()

        # self.dim_half = dim_hidden//2
        # self.dim_vtw = dim_vtw
        self.dim_hidden = dim_hidden
        self.lstm_vtw = nn.LSTM(dim_vtw, dim_hidden, batch_first=True)
        # self.conv1d = nn.Conv1d(dim_hidden, dim_hidden, kernel_size=1)
        # self.layer_norm = nn.LayerNorm(dim_hidden)
        # self.dropout = nn.Dropout(dp), dp

    def forward(self, xw, xn):
        """

        :param xw:
        :param xn: bs, sl, ns+1
        :return:
        """

        bs, sl, _ = xn.size()
        bw = torch.zeros([bs, sl, self.dim_hidden])
        for bi in range(bs):
            max_nw = xn[bi, :, -1].max()    # 最大窗口数
            min_nw = xn[bi, :, -1].min()    # 最大窗口数
            for nw in range(min_nw, max_nw + 1):
                t_indexes = torch.where(xn[bi, :, -1] == nw)[0]
                mbw = xw[bi, t_indexes, :nw, :]  # (mini batch, num vtw, feature)
                _, (h_t, _) = self.lstm_vtw(mbw)  # h_t隐藏层最终输出，(mini batch,seq=nw, hidden)
                bw[bi, t_indexes, :] = h_t.squeeze()
        # bwr = self.conv1d(torch.cat((br, bw), 2).transpose(1, 2)).transpose(1, 2)   # (batch, seq, hidden)
        # # out = self.dropout(self.layer_norm(bwr))  # (batch, seq, hidden)
        # out = self.layer_norm(bwr)
        return bw  # batch, seq, feature


# 1.3 针对单颗卫星的信息嵌入,并与全局信息整合
class AgentEmbedding(nn.Module):
    def __init__(self, dim_vtw, dim_half, dim_hidden):
        super(AgentEmbedding, self).__init__()
        self.dim_half = dim_half
        # self.dim_vtw = dim_vtw
        # self.dim_hidden = dim_hidden
        self.lstm_vtw = nn.LSTM(dim_vtw, dim_half, batch_first=True)

        self.conv1d = nn.Conv1d(dim_hidden, dim_hidden, kernel_size=1)
        self.layer_norm = nn.LayerNorm(dim_hidden)

    def forward(self, xw, xn, er, egw):
        '''

        :param xw: 部分任务在单颗卫星下的窗口,(bs, num_task, num_task, :)
        :param xn: (bs,num_task)
        :param er:
        :param egw:
        :return:
        '''

        bs, sl = xn.size()
        bw = torch.zeros([bs, sl, self.dim_half])   # 没有窗口的就是全零值
        for bi in range(bs):
            max_nw = xn[bi, :].max()  # 最大窗口数
            for nw in range(1, max_nw + 1):
                t_indexes = torch.where(xn[bi, :] == nw)[0]
                mbw = xw[bi, t_indexes, :nw, :]  # (mini batch, num vtw, feature)
                _, (h_t, _) = self.lstm_vtw(mbw)  # h_t隐藏层最终输出，(mini batch,seq=nw, hidden)
                bw[bi, t_indexes, :] = h_t.squeeze()
        bwr = self.conv1d(torch.cat((er, egw+bw), 2).transpose(1, 2)).transpose(1, 2)  # (batch, seq, hidden)
        # # out = self.dropout(self.layer_norm(bwr))  # (batch, seq, hidden)
        bwr = self.layer_norm(bwr)
        return bwr


# 2. 单智能体编码器
# 编码器,由Tansformer的编码器构造,用于提取特征,可以分别提取全局特征和阶段特征
class AgentEncoder(nn.Module):
    def __init__(self, d_model, num_heads, dp):
        super(AgentEncoder, self).__init__()
        self.multihead = nn.MultiheadAttention(d_model, num_heads, dropout=dp)
        # (L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)    # （batch, channel, seq）
        self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        '''

        :param input: bs, sl, f
        :return:
        '''

        input = input.transpose(0, 1)   # bs, sl, f 转换为 sl, bs, f
        output, _ = self.multihead(input, input, input)    # 输入输出都是 sl, bs, f
        input = self.layer_norm((input + output).transpose(0, 1))    # 转换为 bs,sl,f 再 Add&Norm

        output = F.relu(self.conv1(input.transpose(1, 2)))    # # （batch, channel, seq）
        output = self.conv2(output).transpose(1, 2)         # 输出 bs, sl, f
        output = self.layer_norm(input + output)
        return output


# 4. 单智能体解码器-注意力模型,也就是自回归的方式解码
class AgentDecoder(nn.Module):
    def __init__(self, dim_state, dim_hidden):
        super(AgentDecoder, self).__init__()
        self.linear1 = nn.Linear(dim_state, dim_hidden)  # bs, f
        self.linear2 = nn.Linear(dim_hidden * 2, dim_hidden)  # bs, f
        self.lstm_cell = nn.LSTMCell(dim_hidden, dim_hidden)

        self.input_linear = nn.Linear(dim_hidden, dim_hidden)
        self.context_linear = nn.Conv1d(dim_hidden, dim_hidden, kernel_size=1)  # 这个卷积输入是(N,C,L),batch,feature,seq
        self.V = Parameter(torch.FloatTensor(dim_hidden), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, xs, xe, hc, context, mask):
        """
        :param xs: bs,hidden
        :param xe: bs,hidden
        :param hc: ht,ct
        :param context: bs,sl,hidden
        :return:
        """
        xs = self.linear1(xs)
        xse = torch.cat((xs, xe), dim=1)
        xse = F.relu(self.linear2(xse))
        ht, ct = self.lstm_cell(xse, hc)

        context = context.permute(0, 2, 1)  # -> bs,h,sl
        inp = self.input_linear(ht).unsqueeze(2).expand(-1, -1, context.size(2))  # 扩展到序列长度
        ctx = self.context_linear(context)  # 输入(batch, hidden_dim, seq_len)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)    # (batch, 1, hidden_dim)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)  # 1,seq_len
        if len(att[mask]) > 0:      # 存在需要屏蔽的值,真表示要屏蔽
            att[mask] = self.inf[mask]
        alpha = self.softmax(att)   # 注意力分布
        return ht, ct, alpha

    def init_inf(self, mask_size):  # # 输入的mask中真值被屏蔽
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class MultiAgentModel_noLocalFeature(nn.Module):
    def __init__(self, dim_require, dim_vtw, dim_hidden, num_heads, dp=0.1, num_agent=20):     # dim_state,
        super(MultiAgentModel_noLocalFeature, self).__init__()
        self.dim_hidden = dim_hidden
        # dim_half = dim_hidden//2
        self.num_agent = num_agent   # 对应的是最大卫星数
        self.embedding = Embedding(dim_require, dim_vtw, dim_hidden)
        self.encoder = AgentEncoder(dim_hidden, num_heads, dp)
        self.agent_net = []
        dim_state = 6
        for a in range(num_agent):
            agent_decoder = AgentDecoder(dim_state, dim_hidden)
            self.add_module('AgentDecoder_{}'.format(a), agent_decoder)
            self.agent_net.append({'AgentDecoder_{}'.format(a): agent_decoder})

    def forward(self, dr, dw, dn, mo, nr, ngw, sat_arg):    # nlw
        """

        :param dr: 任务需求信息
        :param dw: 任务窗口信息
        :param dn: 任务窗口数量
        :param mo: 任务轨道窗口查询矩阵
        :param nr: 归一化窗口信息
        :param ngw: 归一化全局窗口信息
        :param nlw: 归一化分类窗口信息
        :param sat_arg: 卫星参数
        :return:
        """
        bs, task_num, _ = nr.size()
        emb_out = self.embedding(nr, ngw, dn)
        if torch.isnan(emb_out).any():
            raise ValueError('embedding_require存在nan值!')
        enc_global = self.encoder(emb_out)
        # 用sat表示卫星编号，si表示卫星索引
        if sat_arg.sat_list:     # 列表不为空, 列表中表示实际应用的卫星编号，卫星编号是唯一的，但是输入的数据是连续排列的
            sat_num = len(sat_arg.sat_list)     # 实际的卫星数量
            sat_list = sat_arg.sat_list
        else:
            sat_num = self.num_agent
            sat_list = list(range(sat_num))
        sat_index_list = list(range(sat_num))   # 卫星索引列表

        for si in sat_index_list:
            sat = sat_list[si]
            self.agent_net[sat]['AgentDecoder_{}'.format(sat)].init_inf(torch.Size([bs, task_num]))  # 注意力机制初始化
        # batch_state = torch.empty([bs, sat_num + 1], dtype=bool)    # 样本在各卫星的调度状态，真表示还需要规划，最后一列表示样本的总体状态
        # # 需要初始化，哪些样本在哪些卫星下还可以继续规划, batch_state的初始化和更新不一样
        # # 根据任务在各颗卫星的窗口数量判断该样本是否在该卫星下有调度机会
        # # 先计算每个样本在各颗卫星的窗口总数，>0表示该样本有调度机会
        # batch_state[:, :sat_num] = (dn[:, :, :sat_num].sum(dim=1)).gt(0)
        # # 最后一列表示样本是否完成调度
        # batch_state[:, sat_num] = batch_state[:, :sat_num].sum(dim=1, dtype=bool)
        # batch_task_state = torch.ones([bs, num_task], dtype=bool)   # 表示任务是否待规划

        sat_state = torch.tensor([1, 0, 0, 0, sat_arg.memory, sat_arg.energy], dtype=float).repeat(sat_num, bs, 1)  # 初始状态
        # 0所在轨道（初始为1，表示初始轨道），1空闲开始时间，2俯仰角，3滚动角，4存储，5能量

        h_t = [torch.zeros([bs, self.dim_hidden]) for _ in range(sat_num)]
        c_t = [torch.zeros([bs, self.dim_hidden]) for _ in range(sat_num)]

        sat_task_num = np.zeros([bs, sat_num])
        results = [[[] for _ in range(sat_num)] for _ in range(bs)]
        results_delete = [[[] for _ in range(sat_num)] for _ in range(bs)]
        batch_prob = torch.zeros([bs, task_num])  # 每个动作被选中时的概率

        seq_input = [torch.zeros(torch.Size((bs, self.dim_hidden))) for _ in range(sat_num)]  # new_zeros会复制数据类型和所在设备等
        # 开始标志符
        while mo[:, :, :, :, 0].sum(dim=[3,2,1], dtype=bool).any():
            # any有真为真,表示有样本未调度；没有就退出当前循环
            # d1.确定卫星指针的顺序
            sat_index_list = sat_task_num.mean(axis=0).argsort()    # 按平均完成的任务数升序排列

            # d2.按照顺序逐个自回归
            for si in sat_index_list:
                # 先找出未完成的样本
                ub_set = mo[:, :, si, :, 0].sum(dim=[2, 1]).nonzero(as_tuple=True)[0]   # 未完成规划的样本
                if ub_set.size(0) == 0:     # 没有未完成样本，下一颗卫星
                    continue
                candidate_task_list = [[] for _ in range(bs)]
                sat_task_mask = torch.ones([bs, task_num], dtype=bool)    # 初始化为真，注意力机制中屏蔽真值
                # 更新卫星状态
                for bi in ub_set:
                    # 找出未规划任务,并更新mask矩阵
                    current_orbit = mo[bi, :, si, :, 0].nonzero(as_tuple=True)[1].min()
                    candidate_task_list[bi] = mo[bi, :, si, current_orbit, 0].nonzero(as_tuple=True)[0]
                    sat_task_mask[bi, candidate_task_list[bi]] = False
                    # 更新卫星当前状态
                    if current_orbit > sat_state[si, bi, 0]:
                        sat_state[si, bi, 0] = current_orbit
                        sat_state[si, bi, 4] = sat_arg.memory
                        sat_state[si, bi, 5] = sat_arg.energy

                # 3.1 卫星状态归一化
                norm_state = torch.zeros([bs, 6])
                norm_state[:, 0] = sat_state[si, :, 0].true_divide(sat_arg.orbit_times[si])
                norm_state[:, 1] = sat_state[si, :, 1].true_divide(sat_arg.period)
                norm_state[:, 2] = (sat_state[si, :, 2] - sat_arg.min_pitch).true_divide(sat_arg.max_pitch)
                norm_state[:, 3] = (sat_state[si, :, 3] - sat_arg.min_roll).true_divide(sat_arg.max_roll)
                norm_state[:, 4] = sat_state[si, :, 4].true_divide(sat_arg.memory)
                norm_state[:, 5] = sat_state[si, :, 5].true_divide(sat_arg.energy)

                # 3.1 计算当前指针下的概率分布
                sat = sat_list[si]  # 实际的卫星编号才对应agent编号
                h_t[si], c_t[si], probability = self.agent_net[sat]['AgentDecoder_{}'.format(sat)]\
                    (norm_state, seq_input[si], (h_t[si], c_t[si]), enc_global, sat_task_mask)
                # d3.1下面根据概率分布和卫星状态确定选择的任务，获取单步调度结果
                # (1) 筛选样本，逐个样本计算
                for bi in ub_set:   # 逐个样本处理，计算单步调度结果
                    candidate_tasks = candidate_task_list[bi]  # 总的候选任务集合
                    # 需要注意，是否会因为candidate_tasks导致维度变化
                    step_results, loss_vtw_tasks, sat_state[si, bi, :], \
                        mo[bi, candidate_tasks, si, int(sat_state[si, bi, 0]), :] = self.SingleAgent_StepScheduler(
                        probability[bi, candidate_tasks], sat_state[si, bi, :],
                        mo[bi, candidate_tasks, si, int(sat_state[si, bi, 0]), :], dr[bi, candidate_tasks],
                        dw[bi, candidate_tasks, si, :, :], sat_arg.AngleVelocity, sat_arg.AngleAcceleration,
                        sat_arg.mc_rate, sat_arg.eco_rate, sat_arg.ect_rate) # dn[bi, candidate_tasks, si],
                    if step_results[0] >= 0:    # 表明选中了一个可行任务
                        select_ti = candidate_tasks[int(step_results[0])]  # 实际任务编号
                        # 更新选中任务的后续观测机会状态
                        mo[bi, select_ti, :, :, 0] = 0  # 删除该任务在所有卫星所有轨道上的观测机会
                        # 存入单步调度结果
                        step_results[0] = select_ti
                        results[bi][si].append(step_results.numpy())
                        sat_task_num[bi, si] += 1
                        # 更新概率矩阵 选中任务的概率值
                        batch_prob[bi, select_ti] = probability[bi, select_ti]
                        # 更新seq_input
                        seq_input[si][bi, :] = enc_global[bi, select_ti, :]
                    # 如果没有选中，所有未规划任务在当前卫星下都不会再有观测机会
                    # 卫星根据当前状态无法后续观测的任务，其任务观测机会状态已经在单步调度器中更新过
                    # 还需要判断其在其它卫星是否有调度机会，没有就记录概率并删除
                    if loss_vtw_tasks.size(0) > 0:     # 存在损失观测机会的任务
                        loss_vtw_tasks = candidate_tasks[loss_vtw_tasks]     # 实际任务编号
                        # 根据mo中的值判断, 获得没有后续观测机会的任务, 真表示有后续观测机会
                        dt_state = mo[bi, loss_vtw_tasks, :, :, 0].sum(dim=2).sum(dim=1, dtype=bool)
                        if not dt_state.all():  # 表示有假,即没有后续观测机会了
                            deleted_tasks = loss_vtw_tasks[~dt_state]
                            batch_prob[bi, deleted_tasks] = probability[bi, deleted_tasks]  # 记录其概率
                            results_delete[bi][si] += deleted_tasks.tolist()
                    # 在所有卫星单步调度完成后，更新样本卫星下的状态
                    # 至此一个样本在一个卫星下单步调度完成

            # 至此，所有卫星的单步调度完成，mo没单步调度后就会及时更新
        # 至此，所有样本的所有任务调度完毕
        # 汇总调度结果,计算适应度值
        result_evaluation = torch.zeros([bs, 8])
        # 评价指标：0总任务数，1完成任务数，2任务完成率，3总收益，4完成收益，5任务收益率, 6负载均衡, 7优化目标
        batch_sat_eval = torch.zeros([bs, sat_num, 2])    # 0完成任务数 1完成任务收益之和
        for bi in range(bs):
            for si in range(sat_num):
                results[bi][si] = np.vstack(results[bi][si])
                batch_sat_eval[bi, si, 0] = results[bi][si].shape[0]     # 该卫星完成的任务数
                batch_sat_eval[bi, si, 1] = dr[bi, np.int32(results[bi][si][:, 0]), 1].sum()   # 优先级之和
                # 评价指标：0总任务数，1完成任务数，2任务完成率，3总收益，4完成收益，5任务收益率, 6负载均衡, 7优化目标
        result_evaluation[:, 0] = task_num  # 总任务数
        result_evaluation[:, 1] = batch_sat_eval[:, :, 0].sum(dim=1)
        result_evaluation[:, 2] = result_evaluation[:, 1] / result_evaluation[:, 0]
        result_evaluation[:, 3] = dr[:, :, 1].sum(dim=1)  # 总收益
        result_evaluation[:, 4] = batch_sat_eval[:, :, 1].sum(dim=1)
        result_evaluation[:, 5] = result_evaluation[:, 4] / result_evaluation[:, 3]
        # mean_time = batch_sat_eval[:, :, 0].mean(dim=1)   # 每个样本下平均卫星执行任务数
        result_evaluation[:, 6] = 1/batch_sat_eval[:, :, 0].mean(dim=1) * batch_sat_eval[:, :, 0].std(dim=1)
        result_evaluation[:, 7] = result_evaluation[:, 5] - result_evaluation[:, 6]

        return batch_prob, results, result_evaluation, results_delete

    def SingleAgent_StepScheduler(self, prob, state_SingleSat, state_task_orbit, table_require, table_vtw,
                                  maxV, angA, mr, eor, ear):
        """
        :param prob: 当前指针下的在当前卫星下的未规划任务的概率分布，(num_task)
        :param state_SingleSat:单星状态，(6) # 0所在轨道（初始为1），1空闲开始时间，2俯仰角，3滚动角，4存储，5能量
        :param state_task_orbit:  torch.int32 所有候选任务状态，(num_candidate_task, 2)
                                #0表示未规划任务是否在该轨道上有窗口(0或1), 1表示
        :param table_vtw:  任务的窗口信息 (num_candidate_task, max_nw, 7) 去掉了卫星维度的信息
                0 卫星编号，1 轨道编号（从1开始），2 最早开始时间， 3 最晚开始时间，4 滚动角，5 斜率k，6 截距b
        :param table_require:任务的需求信息，(num_candidate_task, 2) 持续时间和优先级
        :param maxV:最大角速度  :param table_nw: 任务在该卫星下的窗口数 (num_candidate_task)
        :param angA:角加速度
        :param maxM:最大存储
        :param maxE:最大能量
        :param mr:存储消耗
        :param eor:观测能量消耗
        :param ear: 姿态调整能量消耗
        :return:
        """

        scheduling_results = torch.zeros(6) - 1
        # 0选择的动作即任务，1所在轨道，2开始时间，3结束时间，4俯仰角，5滚动角
        candidate_tasks = list(range(prob.size(0)))

        # ss1   当前样本下根据卫星状态局部选择任务
        # set_ut = state_task[bi, :].nonzero().squeeze()  # 非0即为真 未规划的任务集合
        flag_exist_task = False
        all_loss_tasks = []    # 存入所有损失了观测机会的任务索引
        # 候选任务按照概率降序排列
        sort_candidate_tasks = prob.argsort(descending=True)  # 默认降序
        for uti in sort_candidate_tasks:  # 逐个任务判断
            uwi = state_task_orbit[uti, 1]   # 窗口编号
            flag_temp, oa = calculate_EarliestObservationAction(
                state_SingleSat[1:4].tolist(), table_vtw[uti, uwi, 2:].tolist(),
                table_require[uti, 0].tolist(), maxV, angA)
            if flag_temp:   # 满足动作约束，计算存储和能量约束
                conM = table_require[uti, 0] * mr
                conE = table_require[uti, 0] * eor + (abs(oa[2] - state_SingleSat[2]) +
                                                          abs(oa[3] - state_SingleSat[3])) * ear

                if conM <= state_SingleSat[4] and conE <= state_SingleSat[5]:   # 满足存储和能量约束
                    # 更新卫星状态
                    flag_exist_task = True
                    # state_SingleSat[0] = current_orbit  # 轨道是在规划前更新的
                    state_SingleSat[1] = oa[1]  # ft 空闲开始时间
                    state_SingleSat[2] = oa[2]  # 俯仰角
                    state_SingleSat[3] = oa[3]  # 滚动角
                    state_SingleSat[4] = state_SingleSat[4] - conM
                    state_SingleSat[5] = state_SingleSat[5] - conE
                    # 更新任务轨道状态,在函数外部更新
                    # 更新调度方案
                    scheduling_results[0] = uti  # 任务索引，不是任务编号
                    scheduling_results[1] = state_SingleSat[0]
                    scheduling_results[2:] = torch.tensor(oa)
                    # 从候选任务中删除选定的任务
                    candidate_tasks = candidate_tasks[:uti] + candidate_tasks[uti + 1:]
                    break  # 跳出循环
        # 是否存在遍历所有任务，都没有可执行任务
        if flag_exist_task:     # 真表示存在可行任务
            # 需要判断当前候选任务在当前轨道上的观测窗口是否还有效  # ft oa[1]  空闲开始时间
            for uti in candidate_tasks:
                if table_vtw[uti, state_task_orbit[uti, 1], 3] < oa[1]:
                    state_task_orbit[uti, 0] = 0
                    all_loss_tasks.append(uti)
        else:   # 假表示当前轨道上的候选任务都不可行
            # 因为要进入下一个轨道了，当前轨道上任务就会失去观测机会
            state_task_orbit[candidate_tasks, 0] = 0
            all_loss_tasks = candidate_tasks
        # 至此，当前样本的当前轨道候选任务选择结束
        # if len(all_loss_tasks)==0:
        #     print(1)
        # all_loss_tasks = torch.tensor(all_loss_tasks)    # 横向拼接形成所有选择过的候选任务
        # 当前样本单步调度完成
        return scheduling_results, torch.tensor(all_loss_tasks), state_SingleSat, state_task_orbit


class MultiAgentModel_noPointerSort(nn.Module):
    def __init__(self, dim_require, dim_vtw, dim_hidden, num_heads, dp=0.1, num_agent=20):     # dim_state,
        super(MultiAgentModel_noPointerSort, self).__init__()
        self.dim_hidden = dim_hidden
        dim_half = dim_hidden//2
        self.num_agent = num_agent   # 对应的是最大卫星数
        self.require_embedding = RequireEmbedding(dim_require, dim_half)
        self.full_embedding = FullEmbedding(dim_vtw, dim_half)
        self.agent_net = []
        dim_state = 6
        for a in range(num_agent):
            agent_embedding = AgentEmbedding(dim_vtw, dim_half, dim_hidden)
            agent_encoder = AgentEncoder(dim_hidden, num_heads, dp)
            agent_decoder = AgentDecoder(dim_state, dim_hidden)
            self.add_module('AgentEmbedding_{}'.format(a), agent_embedding)
            self.add_module('AgentEncoder_{}'.format(a), agent_encoder)
            self.add_module('AgentDecoder_{}'.format(a), agent_decoder)
            self.agent_net.append({'AgentEmbedding_{}'.format(a): agent_embedding,
                                 'AgentEncoder_{}'.format(a): agent_encoder,
                                 'AgentDecoder_{}'.format(a): agent_decoder})

    def forward(self, dr, dw, dn, mo, nr, ngw, nlw, sat_arg):    # nlw
        """

        :param dr: 任务需求信息
        :param dw: 任务窗口信息
        :param dn: 任务窗口数量
        :param mo: 任务轨道窗口查询矩阵
        :param nr: 归一化窗口信息
        :param ngw: 归一化全局窗口信息
        :param nlw: 归一化分类窗口信息
        :param sat_arg: 卫星参数
        :return:
        """
        bs, task_num, _ = nr.size()
        # emb_out = self.embedding_layer(nr, ngw, dn)
        # enc_global = self.global_encoder(emb_out)
        # del nr, ngw, emb_out
        # 用sat表示卫星编号，si表示卫星索引
        if sat_arg.sat_list:     # 列表不为空, 列表中表示实际应用的卫星编号，卫星编号是唯一的，但是输入的数据是连续排列的
            sat_num = len(sat_arg.sat_list)     # 实际的卫星数量
            sat_list = sat_arg.sat_list
        else:
            sat_num = self.num_agent
            sat_list = list(range(sat_num))
        sat_index_list = list(range(sat_num))   # 卫星索引列表

        agent_encoding = [[] for _ in range(sat_num)]
        embedding_require = self.require_embedding(nr)
        if torch.isnan(embedding_require).any():
            raise ValueError('embedding_require存在nan值!')
        embedding_gw = self.full_embedding(ngw, dn)

        for si in sat_index_list:
            sat = sat_list[si]
            agent_embedding = self.agent_net[sat]['AgentEmbedding_{}'.format(sat)](nlw[:, :, si, :, :], dn[:, :, si], embedding_require, embedding_gw)
            agent_encoding[si] = self.agent_net[sat]['AgentEncoder_{}'.format(sat)](agent_embedding)
            self.agent_net[sat]['AgentDecoder_{}'.format(sat)].init_inf(torch.Size([bs, task_num]))  # 注意力机制初始化
        # batch_state = torch.empty([bs, sat_num + 1], dtype=bool)    # 样本在各卫星的调度状态，真表示还需要规划，最后一列表示样本的总体状态
        # # 需要初始化，哪些样本在哪些卫星下还可以继续规划, batch_state的初始化和更新不一样
        # # 根据任务在各颗卫星的窗口数量判断该样本是否在该卫星下有调度机会
        # # 先计算每个样本在各颗卫星的窗口总数，>0表示该样本有调度机会
        # batch_state[:, :sat_num] = (dn[:, :, :sat_num].sum(dim=1)).gt(0)
        # # 最后一列表示样本是否完成调度
        # batch_state[:, sat_num] = batch_state[:, :sat_num].sum(dim=1, dtype=bool)
        # batch_task_state = torch.ones([bs, num_task], dtype=bool)   # 表示任务是否待规划

        sat_state = torch.tensor([1, 0, 0, 0, sat_arg.memory, sat_arg.energy], dtype=float).repeat(sat_num, bs, 1)  # 初始状态
        # 0所在轨道（初始为1，表示初始轨道），1空闲开始时间，2俯仰角，3滚动角，4存储，5能量

        h_t = [torch.zeros([bs, self.dim_hidden]) for _ in range(sat_num)]
        c_t = [torch.zeros([bs, self.dim_hidden]) for _ in range(sat_num)]

        sat_task_num = np.zeros([bs, sat_num])
        results = [[[] for _ in range(sat_num)] for _ in range(bs)]
        results_delete = [[[] for _ in range(sat_num)] for _ in range(bs)]
        batch_prob = torch.zeros([bs, task_num])  # 每个动作被选中时的概率

        seq_input = [torch.zeros(torch.Size((bs, self.dim_hidden))) for _ in range(sat_num)]  # new_zeros会复制数据类型和所在设备等
        # 开始标志符
        while mo[:, :, :, :, 0].sum(dim=[3,2,1], dtype=bool).any():
            # any有真为真,表示有样本未调度；没有就退出当前循环
            # d1.确定卫星指针的顺序
            # sat_index_list = sat_task_num.mean(axis=0).argsort()    # 按平均完成的任务数升序排列

            # d2.按照顺序逐个自回归
            for si in sat_index_list:
                # 先找出未完成的样本
                ub_set = mo[:, :, si, :, 0].sum(dim=[2, 1]).nonzero(as_tuple=True)[0]   # 未完成规划的样本
                if ub_set.size(0) == 0:     # 没有未完成样本，下一颗卫星
                    continue
                candidate_task_list = [[] for _ in range(bs)]
                sat_task_mask = torch.ones([bs, task_num], dtype=bool)    # 初始化为真，注意力机制中屏蔽真值
                # 更新卫星状态
                for bi in ub_set:
                    # 找出未规划任务,并更新mask矩阵
                    current_orbit = mo[bi, :, si, :, 0].nonzero(as_tuple=True)[1].min()
                    candidate_task_list[bi] = mo[bi, :, si, current_orbit, 0].nonzero(as_tuple=True)[0]
                    sat_task_mask[bi, candidate_task_list[bi]] = False
                    # 更新卫星当前状态
                    if current_orbit > sat_state[si, bi, 0]:
                        sat_state[si, bi, 0] = current_orbit
                        sat_state[si, bi, 4] = sat_arg.memory
                        sat_state[si, bi, 5] = sat_arg.energy

                # 3.1 卫星状态归一化
                norm_state = torch.zeros([bs, 6])
                norm_state[:, 0] = sat_state[si, :, 0].true_divide(sat_arg.orbit_times[si])
                norm_state[:, 1] = sat_state[si, :, 1].true_divide(sat_arg.period)
                norm_state[:, 2] = (sat_state[si, :, 2] - sat_arg.min_pitch).true_divide(sat_arg.max_pitch)
                norm_state[:, 3] = (sat_state[si, :, 3] - sat_arg.min_roll).true_divide(sat_arg.max_roll)
                norm_state[:, 4] = sat_state[si, :, 4].true_divide(sat_arg.memory)
                norm_state[:, 5] = sat_state[si, :, 5].true_divide(sat_arg.energy)

                # 3.1 计算当前指针下的概率分布
                sat = sat_list[si]  # 实际的卫星编号才对应agent编号
                h_t[si], c_t[si], probability = self.agent_net[sat]['AgentDecoder_{}'.format(sat)]\
                    (norm_state, seq_input[si], (h_t[si], c_t[si]), agent_encoding[si], sat_task_mask)
                # d3.1下面根据概率分布和卫星状态确定选择的任务，获取单步调度结果
                # (1) 筛选样本，逐个样本计算
                for bi in ub_set:   # 逐个样本处理，计算单步调度结果
                    candidate_tasks = candidate_task_list[bi]  # 总的候选任务集合
                    # 需要注意，是否会因为candidate_tasks导致维度变化
                    step_results, loss_vtw_tasks, sat_state[si, bi, :], \
                        mo[bi, candidate_tasks, si, int(sat_state[si, bi, 0]), :] = self.SingleAgent_StepScheduler(
                        probability[bi, candidate_tasks], sat_state[si, bi, :],
                        mo[bi, candidate_tasks, si, int(sat_state[si, bi, 0]), :], dr[bi, candidate_tasks],
                        dw[bi, candidate_tasks, si, :, :], sat_arg.AngleVelocity, sat_arg.AngleAcceleration,
                        sat_arg.mc_rate, sat_arg.eco_rate, sat_arg.ect_rate) # dn[bi, candidate_tasks, si],
                    if step_results[0] >= 0:    # 表明选中了一个可行任务
                        select_ti = candidate_tasks[int(step_results[0])]  # 实际任务编号
                        # 更新选中任务的后续观测机会状态
                        mo[bi, select_ti, :, :, 0] = 0  # 删除该任务在所有卫星所有轨道上的观测机会
                        # 存入单步调度结果
                        step_results[0] = select_ti
                        results[bi][si].append(step_results.numpy())
                        sat_task_num[bi, si] += 1
                        # 更新概率矩阵 选中任务的概率值
                        batch_prob[bi, select_ti] = probability[bi, select_ti]
                        # 更新seq_input
                        seq_input[si][bi, :] = agent_encoding[si][bi, select_ti, :]
                    # 如果没有选中，所有未规划任务在当前卫星下都不会再有观测机会
                    # 卫星根据当前状态无法后续观测的任务，其任务观测机会状态已经在单步调度器中更新过
                    # 还需要判断其在其它卫星是否有调度机会，没有就记录概率并删除
                    if loss_vtw_tasks.size(0) > 0:     # 存在损失观测机会的任务
                        loss_vtw_tasks = candidate_tasks[loss_vtw_tasks]     # 实际任务编号
                        # 根据mo中的值判断, 获得没有后续观测机会的任务, 真表示有后续观测机会
                        dt_state = mo[bi, loss_vtw_tasks, :, :, 0].sum(dim=2).sum(dim=1, dtype=bool)
                        if not dt_state.all():  # 表示有假,即没有后续观测机会了
                            deleted_tasks = loss_vtw_tasks[~dt_state]
                            batch_prob[bi, deleted_tasks] = probability[bi, deleted_tasks]  # 记录其概率
                            results_delete[bi][si] += deleted_tasks.tolist()
                    # 在所有卫星单步调度完成后，更新样本卫星下的状态
                    # 至此一个样本在一个卫星下单步调度完成

            # 至此，所有卫星的单步调度完成，mo没单步调度后就会及时更新
        # 至此，所有样本的所有任务调度完毕
        # 汇总调度结果,计算适应度值
        result_evaluation = torch.zeros([bs, 8])
        # 评价指标：0总任务数，1完成任务数，2任务完成率，3总收益，4完成收益，5任务收益率, 6负载均衡, 7优化目标
        batch_sat_eval = torch.zeros([bs, sat_num, 2])    # 0完成任务数 1完成任务收益之和
        for bi in range(bs):
            for si in range(sat_num):
                results[bi][si] = np.vstack(results[bi][si])
                batch_sat_eval[bi, si, 0] = results[bi][si].shape[0]     # 该卫星完成的任务数
                batch_sat_eval[bi, si, 1] = dr[bi, np.int32(results[bi][si][:, 0]), 1].sum()   # 优先级之和
                # 评价指标：0总任务数，1完成任务数，2任务完成率，3总收益，4完成收益，5任务收益率, 6负载均衡, 7优化目标
        result_evaluation[:, 0] = task_num  # 总任务数
        result_evaluation[:, 1] = batch_sat_eval[:, :, 0].sum(dim=1)
        result_evaluation[:, 2] = result_evaluation[:, 1] / result_evaluation[:, 0]
        result_evaluation[:, 3] = dr[:, :, 1].sum(dim=1)  # 总收益
        result_evaluation[:, 4] = batch_sat_eval[:, :, 1].sum(dim=1)
        result_evaluation[:, 5] = result_evaluation[:, 4] / result_evaluation[:, 3]
        # mean_time = batch_sat_eval[:, :, 0].mean(dim=1)   # 每个样本下平均卫星执行任务数
        result_evaluation[:, 6] = 1/batch_sat_eval[:, :, 0].mean(dim=1) * batch_sat_eval[:, :, 0].std(dim=1)
        result_evaluation[:, 7] = result_evaluation[:, 5] - result_evaluation[:, 6]

        return batch_prob, results, result_evaluation, results_delete

    def SingleAgent_StepScheduler(self, prob, state_SingleSat, state_task_orbit, table_require, table_vtw,
                                  maxV, angA, mr, eor, ear):
        """
        :param prob: 当前指针下的在当前卫星下的未规划任务的概率分布，(num_task)
        :param state_SingleSat:单星状态，(6) # 0所在轨道（初始为1），1空闲开始时间，2俯仰角，3滚动角，4存储，5能量
        :param state_task_orbit:  torch.int32 所有候选任务状态，(num_candidate_task, 2)
                                #0表示未规划任务是否在该轨道上有窗口(0或1), 1表示
        :param table_vtw:  任务的窗口信息 (num_candidate_task, max_nw, 7) 去掉了卫星维度的信息
                0 卫星编号，1 轨道编号（从1开始），2 最早开始时间， 3 最晚开始时间，4 滚动角，5 斜率k，6 截距b
        :param table_require:任务的需求信息，(num_candidate_task, 2) 持续时间和优先级
        :param maxV:最大角速度  :param table_nw: 任务在该卫星下的窗口数 (num_candidate_task)
        :param angA:角加速度
        :param mr:存储消耗
        :param eor:观测能量消耗
        :param ear: 姿态调整能量消耗
        :return:
        """

        scheduling_results = torch.zeros(6) - 1
        # 0选择的动作即任务，1所在轨道，2开始时间，3结束时间，4俯仰角，5滚动角
        candidate_tasks = list(range(prob.size(0)))

        # ss1   当前样本下根据卫星状态局部选择任务
        # set_ut = state_task[bi, :].nonzero().squeeze()  # 非0即为真 未规划的任务集合
        flag_exist_task = False
        all_loss_tasks = []    # 存入所有损失了观测机会的任务索引
        # 候选任务按照概率降序排列
        sort_candidate_tasks = prob.argsort(descending=True)  # 默认降序
        for uti in sort_candidate_tasks:  # 逐个任务判断
            uwi = state_task_orbit[uti, 1]   # 窗口编号
            flag_temp, oa = calculate_EarliestObservationAction(
                state_SingleSat[1:4].tolist(), table_vtw[uti, uwi, 2:].tolist(),
                table_require[uti, 0].tolist(), maxV, angA)
            if flag_temp:   # 满足动作约束，计算存储和能量约束
                conM = table_require[uti, 0] * mr
                conE = table_require[uti, 0] * eor + (abs(oa[2] - state_SingleSat[2]) +
                                                          abs(oa[3] - state_SingleSat[3])) * ear

                if conM <= state_SingleSat[4] and conE <= state_SingleSat[5]:   # 满足存储和能量约束
                    # 更新卫星状态
                    flag_exist_task = True
                    # state_SingleSat[0] = current_orbit  # 轨道是在规划前更新的
                    state_SingleSat[1] = oa[1]  # ft 空闲开始时间
                    state_SingleSat[2] = oa[2]  # 俯仰角
                    state_SingleSat[3] = oa[3]  # 滚动角
                    state_SingleSat[4] = state_SingleSat[4] - conM
                    state_SingleSat[5] = state_SingleSat[5] - conE
                    # 更新任务轨道状态,在函数外部更新
                    # 更新调度方案
                    scheduling_results[0] = uti  # 任务索引，不是任务编号
                    scheduling_results[1] = state_SingleSat[0]
                    scheduling_results[2:] = torch.tensor(oa)
                    # 从候选任务中删除选定的任务
                    candidate_tasks = candidate_tasks[:uti] + candidate_tasks[uti + 1:]
                    break  # 跳出循环
        # 是否存在遍历所有任务，都没有可执行任务
        if flag_exist_task:     # 真表示存在可行任务
            # 需要判断当前候选任务在当前轨道上的观测窗口是否还有效  # ft oa[1]  空闲开始时间
            for uti in candidate_tasks:
                if table_vtw[uti, state_task_orbit[uti, 1], 3] < oa[1]:
                    state_task_orbit[uti, 0] = 0
                    all_loss_tasks.append(uti)
        else:   # 假表示当前轨道上的候选任务都不可行
            # 因为要进入下一个轨道了，当前轨道上任务就会失去观测机会
            state_task_orbit[candidate_tasks, 0] = 0
            all_loss_tasks = candidate_tasks
        # 至此，当前样本的当前轨道候选任务选择结束
        # if len(all_loss_tasks)==0:
        #     print(1)
        # all_loss_tasks = torch.tensor(all_loss_tasks)    # 横向拼接形成所有选择过的候选任务
        # 当前样本单步调度完成
        return scheduling_results, torch.tensor(all_loss_tasks), state_SingleSat, state_task_orbit


class MultiAgentModel_Greedy(nn.Module):
    def __init__(self, dim_require, dim_vtw, dim_hidden, num_heads, dp=0.1, num_agent=20):     # dim_state,
        super(MultiAgentModel_Greedy, self).__init__()
        self.dim_hidden = dim_hidden
        dim_half = dim_hidden//2
        self.num_agent = num_agent   # 对应的是最大卫星数
        self.require_embedding = RequireEmbedding(dim_require, dim_half)
        self.full_embedding = FullEmbedding(dim_vtw, dim_half)
        self.agent_net = []
        dim_state = 6
        for a in range(num_agent):
            agent_embedding = AgentEmbedding(dim_vtw, dim_half, dim_hidden)
            agent_encoder = AgentEncoder(dim_hidden, num_heads, dp)
            agent_decoder = AgentDecoder(dim_state, dim_hidden)
            self.add_module('AgentEmbedding_{}'.format(a), agent_embedding)
            self.add_module('AgentEncoder_{}'.format(a), agent_encoder)
            self.add_module('AgentDecoder_{}'.format(a), agent_decoder)
            self.agent_net.append({'AgentEmbedding_{}'.format(a): agent_embedding,
                                 'AgentEncoder_{}'.format(a): agent_encoder,
                                 'AgentDecoder_{}'.format(a): agent_decoder})

    def forward(self, dr, dw, dn, mo, nr, ngw, nlw, sat_arg):    # nlw
        """

        :param dr: 任务需求信息
        :param dw: 任务窗口信息
        :param dn: 任务窗口数量
        :param mo: 任务轨道窗口查询矩阵
        :param nr: 归一化窗口信息
        :param ngw: 归一化全局窗口信息
        :param nlw: 归一化分类窗口信息
        :param sat_arg: 卫星参数
        :return:
        """
        bs, task_num, _ = nr.size()
        # emb_out = self.embedding_layer(nr, ngw, dn)
        # enc_global = self.global_encoder(emb_out)
        # del nr, ngw, emb_out
        # 用sat表示卫星编号，si表示卫星索引
        if sat_arg.sat_list:     # 列表不为空, 列表中表示实际应用的卫星编号，卫星编号是唯一的，但是输入的数据是连续排列的
            sat_num = len(sat_arg.sat_list)     # 实际的卫星数量
            sat_list = sat_arg.sat_list
        else:
            sat_num = self.num_agent
            sat_list = list(range(sat_num))
        sat_index_list = list(range(sat_num))   # 卫星索引列表

        agent_encoding = [[] for _ in range(sat_num)]
        embedding_require = self.require_embedding(nr)
        if torch.isnan(embedding_require).any():
            raise ValueError('embedding_require存在nan值!')
        embedding_gw = self.full_embedding(ngw, dn)

        for si in sat_index_list:
            sat = sat_list[si]
            agent_embedding = self.agent_net[sat]['AgentEmbedding_{}'.format(sat)](nlw[:, :, si, :, :], dn[:, :, si], embedding_require, embedding_gw)
            agent_encoding[si] = self.agent_net[sat]['AgentEncoder_{}'.format(sat)](agent_embedding)
            self.agent_net[sat]['AgentDecoder_{}'.format(sat)].init_inf(torch.Size([bs, task_num]))  # 注意力机制初始化
        # batch_state = torch.empty([bs, sat_num + 1], dtype=bool)    # 样本在各卫星的调度状态，真表示还需要规划，最后一列表示样本的总体状态
        # # 需要初始化，哪些样本在哪些卫星下还可以继续规划, batch_state的初始化和更新不一样
        # # 根据任务在各颗卫星的窗口数量判断该样本是否在该卫星下有调度机会
        # # 先计算每个样本在各颗卫星的窗口总数，>0表示该样本有调度机会
        # batch_state[:, :sat_num] = (dn[:, :, :sat_num].sum(dim=1)).gt(0)
        # # 最后一列表示样本是否完成调度
        # batch_state[:, sat_num] = batch_state[:, :sat_num].sum(dim=1, dtype=bool)
        # batch_task_state = torch.ones([bs, num_task], dtype=bool)   # 表示任务是否待规划

        sat_state = torch.tensor([1, 0, 0, 0, sat_arg.memory, sat_arg.energy], dtype=float).repeat(sat_num, bs, 1)  # 初始状态
        # 0所在轨道（初始为1，表示初始轨道），1空闲开始时间，2俯仰角，3滚动角，4存储，5能量

        h_t = [torch.zeros([bs, self.dim_hidden]) for _ in range(sat_num)]
        c_t = [torch.zeros([bs, self.dim_hidden]) for _ in range(sat_num)]

        sat_task_num = np.zeros([bs, sat_num])
        results = [[[] for _ in range(sat_num)] for _ in range(bs)]
        results_delete = [[[] for _ in range(sat_num)] for _ in range(bs)]
        batch_prob = torch.zeros([bs, task_num])  # 每个动作被选中时的概率

        seq_input = [torch.zeros(torch.Size((bs, self.dim_hidden))) for _ in range(sat_num)]  # new_zeros会复制数据类型和所在设备等
        # 开始标志符
        while mo[:, :, :, :, 0].sum(dim=[3,2,1], dtype=bool).any():
            # any有真为真,表示有样本未调度；没有就退出当前循环
            # d1.确定卫星指针的顺序
            sat_index_list = sat_task_num.mean(axis=0).argsort()    # 按平均完成的任务数升序排列

            # d2.按照顺序逐个自回归
            for si in sat_index_list:
                # 先找出未完成的样本
                ub_set = mo[:, :, si, :, 0].sum(dim=[2, 1]).nonzero(as_tuple=True)[0]   # 未完成规划的样本
                if ub_set.size(0) == 0:     # 没有未完成样本，下一颗卫星
                    continue
                candidate_task_list = [[] for _ in range(bs)]
                sat_task_mask = torch.ones([bs, task_num], dtype=bool)    # 初始化为真，注意力机制中屏蔽真值
                # 更新卫星状态
                for bi in ub_set:
                    # 找出未规划任务,并更新mask矩阵
                    current_orbit = mo[bi, :, si, :, 0].nonzero(as_tuple=True)[1].min()
                    candidate_task_list[bi] = mo[bi, :, si, current_orbit, 0].nonzero(as_tuple=True)[0]
                    sat_task_mask[bi, candidate_task_list[bi]] = False
                    # 更新卫星当前状态
                    if current_orbit > sat_state[si, bi, 0]:
                        sat_state[si, bi, 0] = current_orbit
                        sat_state[si, bi, 4] = sat_arg.memory
                        sat_state[si, bi, 5] = sat_arg.energy

                # 3.1 卫星状态归一化
                norm_state = torch.zeros([bs, 6])
                norm_state[:, 0] = sat_state[si, :, 0].true_divide(sat_arg.orbit_times[si])
                norm_state[:, 1] = sat_state[si, :, 1].true_divide(sat_arg.period)
                norm_state[:, 2] = (sat_state[si, :, 2] - sat_arg.min_pitch).true_divide(sat_arg.max_pitch)
                norm_state[:, 3] = (sat_state[si, :, 3] - sat_arg.min_roll).true_divide(sat_arg.max_roll)
                norm_state[:, 4] = sat_state[si, :, 4].true_divide(sat_arg.memory)
                norm_state[:, 5] = sat_state[si, :, 5].true_divide(sat_arg.energy)

                # 3.1 计算当前指针下的概率分布
                sat = sat_list[si]  # 实际的卫星编号才对应agent编号
                h_t[si], c_t[si], probability = self.agent_net[sat]['AgentDecoder_{}'.format(sat)]\
                    (norm_state, seq_input[si], (h_t[si], c_t[si]), agent_encoding[si], sat_task_mask)
                # d3.1下面根据概率分布和卫星状态确定选择的任务，获取单步调度结果
                # (1) 筛选样本，逐个样本计算
                for bi in ub_set:   # 逐个样本处理，计算单步调度结果
                    candidate_tasks = candidate_task_list[bi]  # 总的候选任务集合
                    # 需要注意，是否会因为candidate_tasks导致维度变化
                    step_results, loss_vtw_tasks, sat_state[si, bi, :], \
                        mo[bi, candidate_tasks, si, int(sat_state[si, bi, 0]), :] = self.SingleAgent_StepScheduler(
                        probability[bi, candidate_tasks], sat_state[si, bi, :],
                        mo[bi, candidate_tasks, si, int(sat_state[si, bi, 0]), :], dr[bi, candidate_tasks],
                        dw[bi, candidate_tasks, si, :, :], sat_arg.AngleVelocity, sat_arg.AngleAcceleration,
                        sat_arg.memory, sat_arg.energy, sat_arg.mc_rate, sat_arg.eco_rate, sat_arg.ect_rate) # dn[bi, candidate_tasks, si],
                    select_ti = candidate_tasks[int(step_results[0])]  # 实际任务编号
                    # 更新选中任务的后续观测机会状态
                    mo[bi, select_ti, :, :, 0] = 0  # 删除该任务在所有卫星所有轨道上的观测机会
                    if step_results[1] >= 0:    # 表明调度成功
                        # 存入单步调度结果
                        step_results[0] = select_ti
                        results[bi][si].append(step_results.numpy())
                        sat_task_num[bi, si] += 1
                        # 更新概率矩阵 选中任务的概率值
                        batch_prob[bi, select_ti] = probability[bi, select_ti]
                        # 更新seq_input
                        seq_input[si][bi, :] = agent_encoding[si][bi, select_ti, :]
                    # 如果没有选中，所有未规划任务在当前卫星下都不会再有观测机会
                    # 卫星根据当前状态无法后续观测的任务，其任务观测机会状态已经在单步调度器中更新过
                    # 还需要判断其在其它卫星是否有调度机会，没有就记录概率并删除
                    if loss_vtw_tasks.size(0) > 0:     # 存在损失观测机会的任务
                        loss_vtw_tasks = candidate_tasks[loss_vtw_tasks]     # 实际任务编号
                        # 根据mo中的值判断, 获得没有后续观测机会的任务, 真表示有后续观测机会
                        dt_state = mo[bi, loss_vtw_tasks, :, :, 0].sum(dim=2).sum(dim=1, dtype=bool)
                        if not dt_state.all():  # 表示有假,即没有后续观测机会了
                            deleted_tasks = loss_vtw_tasks[~dt_state]
                            batch_prob[bi, deleted_tasks] = probability[bi, deleted_tasks]  # 记录其概率
                            results_delete[bi][si] += deleted_tasks.tolist()
                    # 在所有卫星单步调度完成后，更新样本卫星下的状态
                    # 至此一个样本在一个卫星下单步调度完成

            # 至此，所有卫星的单步调度完成，mo没单步调度后就会及时更新
        # 至此，所有样本的所有任务调度完毕
        # 汇总调度结果,计算适应度值
        result_evaluation = torch.zeros([bs, 8])
        # 评价指标：0总任务数，1完成任务数，2任务完成率，3总收益，4完成收益，5任务收益率, 6负载均衡, 7优化目标
        batch_sat_eval = torch.zeros([bs, sat_num, 2])    # 0完成任务数 1完成任务收益之和
        for bi in range(bs):
            for si in range(sat_num):
                results[bi][si] = np.vstack(results[bi][si])
                batch_sat_eval[bi, si, 0] = results[bi][si].shape[0]     # 该卫星完成的任务数
                batch_sat_eval[bi, si, 1] = dr[bi, np.int32(results[bi][si][:, 0]), 1].sum()   # 优先级之和
                # 评价指标：0总任务数，1完成任务数，2任务完成率，3总收益，4完成收益，5任务收益率, 6负载均衡, 7优化目标
        result_evaluation[:, 0] = task_num  # 总任务数
        result_evaluation[:, 1] = batch_sat_eval[:, :, 0].sum(dim=1)
        result_evaluation[:, 2] = result_evaluation[:, 1] / result_evaluation[:, 0]
        result_evaluation[:, 3] = dr[:, :, 1].sum(dim=1)  # 总收益
        result_evaluation[:, 4] = batch_sat_eval[:, :, 1].sum(dim=1)
        result_evaluation[:, 5] = result_evaluation[:, 4] / result_evaluation[:, 3]
        # mean_time = batch_sat_eval[:, :, 0].mean(dim=1)   # 每个样本下平均卫星执行任务数
        result_evaluation[:, 6] = 1/batch_sat_eval[:, :, 0].mean(dim=1) * batch_sat_eval[:, :, 0].std(dim=1)
        result_evaluation[:, 7] = result_evaluation[:, 5] - result_evaluation[:, 6]

        return batch_prob, results, result_evaluation, results_delete

    def SingleAgent_StepScheduler(self, prob, state_SingleSat, state_task_orbit, table_require, table_vtw,
                                  maxV, angA, maxM, maxE, mr, eor, ear):
        """
        :param prob: 当前指针下的在当前卫星下的未规划任务的概率分布，(num_task)
        :param state_SingleSat:单星状态，(6) # 0所在轨道（初始为1），1空闲开始时间，2俯仰角，3滚动角，4存储，5能量
        :param state_task_orbit:  torch.int32 所有候选任务状态，(num_candidate_task, 2)
                                #0表示未规划任务是否在该轨道上有窗口(0或1), 1表示
        :param table_vtw:  任务的窗口信息 (num_candidate_task, max_nw, 7) 去掉了卫星维度的信息
                0 卫星编号，1 轨道编号（从1开始），2 最早开始时间， 3 最晚开始时间，4 滚动角，5 斜率k，6 截距b
        :param table_require:任务的需求信息，(num_candidate_task, 2) 持续时间和优先级
        :param maxV:最大角速度  :param table_nw: 任务在该卫星下的窗口数 (num_candidate_task)
        :param angA:角加速度
        :param maxM:最大存储
        :param maxE:最大能量
        :param mr:存储消耗
        :param eor:观测能量消耗
        :param ear: 姿态调整能量消耗
        :return:
        """

        scheduling_results = torch.zeros(6) - 1
        # 0选择的动作即任务，1所在轨道，2开始时间，3结束时间，4俯仰角，5滚动角
        candidate_tasks = list(range(prob.size(0)))

        # ss1   当前样本下根据卫星状态局部选择任务
        # set_ut = state_task[bi, :].nonzero().squeeze()  # 非0即为真 未规划的任务集合
        flag_exist_task = False
        all_loss_tasks = []    # 存入所有损失了观测机会的任务索引
        # 候选任务按照概率降序排列
        # sort_candidate_tasks = prob.argsort(descending=True)  # 默认降序
        uti = prob.argmax()     # 选择概率最大的
        scheduling_results[0] = uti  # 任务索引，不是任务编号
        uwi = state_task_orbit[uti, 1]   # 窗口编号
        flag_temp, oa = calculate_EarliestObservationAction(
            state_SingleSat[1:4].tolist(), table_vtw[uti, uwi, 2:].tolist(),
            table_require[uti, 0].tolist(), maxV, angA)
        if flag_temp:   # 满足动作约束，计算存储和能量约束
            conM = table_require[uti, 0] * mr
            conE = table_require[uti, 0] * eor + (abs(oa[2] - state_SingleSat[2]) +
                                                      abs(oa[3] - state_SingleSat[3])) * ear

            if conM <= state_SingleSat[4] and conE <= state_SingleSat[5]:   # 满足存储和能量约束
                # 更新卫星状态
                flag_exist_task = True
                # state_SingleSat[0] = current_orbit  # 轨道是在规划前更新的
                state_SingleSat[1] = oa[1]  # ft 空闲开始时间
                state_SingleSat[2] = oa[2]  # 俯仰角
                state_SingleSat[3] = oa[3]  # 滚动角
                state_SingleSat[4] = state_SingleSat[4] - conM
                state_SingleSat[5] = state_SingleSat[5] - conE
                # 更新任务轨道状态,在函数外部更新
                # 更新调度方案
                scheduling_results[1] = state_SingleSat[0]
                scheduling_results[2:] = torch.tensor(oa)
        # 从候选任务中删除选定的任务
        candidate_tasks = candidate_tasks[:uti] + candidate_tasks[uti + 1:]
        # 是否存在遍历所有任务，都没有可执行任务
        if flag_exist_task:     # 真表示存在可行任务
            # 需要判断当前候选任务在当前轨道上的观测窗口是否还有效  # ft oa[1]  空闲开始时间
            for uti in candidate_tasks:
                if table_vtw[uti, state_task_orbit[uti, 1], 3] < oa[1]:
                    state_task_orbit[uti, 0] = 0
                    all_loss_tasks.append(uti)
        # else:   # 假表示当前选中的候选任务不可行
        #     all_loss_tasks.append(uti)
            # 没有调度成功state_SingleSat, state_task_orbit不会改变
        # 至此，当前样本的当前轨道候选任务选择结束
        # 当前样本单步调度完成
        return scheduling_results, torch.tensor(all_loss_tasks), state_SingleSat, state_task_orbit


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import argparse
    import os
    import time
    import datetime
    from torch.utils.tensorboard import SummaryWriter
    import matplotlib.pyplot as plt

    # 0 明确当前数据的保存路径
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

