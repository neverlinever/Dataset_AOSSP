import numpy as np

# 3.1 单个方向的姿态转换时间的计算
def transT(delt_a, max_v, a):
    """
    # max_v = 3  # 最大角速度
    # a = 1  # 加速度为2 °/s^2
    """

    if delt_a == 0:
        return 0
    elif 0 < delt_a <= max_v**2/a:
        tran_t = 2 * pow(delt_a / a, 0.5)  # 转换时间
    elif delt_a <= 90:  # 最大转换角度就是90度
        tran_t = delt_a / max_v + max_v / a
    else:
        print('角度超出范围')
        return 0
    return tran_t


# 3.2 根据上一个状态和当前待规划任务的信息，计算当前任务的执行动作
# 二次不等式求解
def solve_quadratic_inequality(a, b, c):
    # 解一元二次不等式，可知曲线开口向上，求>=0的范围，所以必然是有解的
    # A>0
    drt = b ** 2 - 4 * a * c
    # is_existed = 1 # 存在解的个数
    outcome = []

    if drt <= 0:  # 表示恒成立
        is_existed = 1
    elif drt > 0:
        x1 = (-b - drt ** 0.5) / (2 * a)  # 小的解
        x2 = (-b + drt ** 0.5) / (2 * a)  # 大的解
        is_existed = 0
        outcome = [x1, x2]
    return is_existed, outcome


# 线性不等式求解
def solve_linear_inequality(a, b):
    # 解线性不等式
    # ax>b
    outcome = []
    if a == 0:  # 无解
        flag = -1
    elif a > 0:  # >outcome
        flag = 1
        outcome = b / a
    else:  # a<0，  <outcome
        flag = 0
        outcome = b / a
    return flag, outcome


# 20240521 把角度作为自变量计算符合条件的最大角，找出最早开始时间
def calculate_EarliestObservationAction(state, vtw, d, V, w, ST=1):
    '''
    根据最早策略计算可行的最大俯仰角对应的最早开始时间，输入的数据类型都是列表
    :param state: 卫星状态 0ft,1pa,2ra
    :param vtw: 时间窗信息 0est,1lst,2ra,3k,4b
    :param d: 持续时间
    :param V: 最大角速度
    :param w: 角加速度
    :param ST: 稳定时间，默认为1
    :return: flag, obs_act  标志符和观测动作
    '''

    k = vtw[3]
    b = vtw[4]
    ft = state[0]
    p0 = state[1]
    deltTR = transT(abs(vtw[2]-state[2]), V, w)

    if k >= 0:
        print('k>=0')
    p_min = -45
    p_max = min((-b+ft+deltTR+ST)/k, 45)

    if p_max < p_min:
        return False, []
    # (1) p1-p0>=V^2/w => p1>=p0+V^2/w  解线性不等式 ax>b
    p_temp = p0 + V**2/w
    p_range = [max(p_temp, p_min), p_max]
    if p_range[0] <= p_range[1]:  # p_min1 > p_max 无解
        is_ge, value = solve_linear_inequality(a=k*V-1, b=(-b+ft+deltTR+ST)*V-p0+V**2/w)
        if is_ge == 0:  # p0+V^2/w<=p1<=value
            if value >= p_range[0]:
                p1 = min(value, p_max)
                st1 = k * p1 + b
                return True, [st1, st1+d, p1, vtw[2]]   # 观测动作
        else:
            print('Error-calculate_EarliestObservationAction 输入不正确')

    # (2) 0<=p1-p0<=V**2/w => -45 <= p0<=p1<=p0+V**2/w=p_temp p_max
    p_range = [max(p_min, p0), min(p_max, p_temp)]
    if p_range[0] <= p_range[1]:
        is_exist, values = solve_quadratic_inequality(a=k**2, b=2*k*(b-ft-deltTR-ST)-4/w, c=(b-ft-deltTR-ST)**2+4/w*p0)
        if is_exist == 1:   # 恒成立,取最大值
            p1 = p_range[1]
            st1 = k * p1 + b
            return True, [st1, st1 + d, p1, vtw[2]]  # 观测动作
        else:
            if p_range[0] <= values[1] <= p_range[1]:
                p1 = p_range[1]
                st1 = k * p1 + b
                return True, [st1, st1 + d, p1, vtw[2]]  # 观测动作
            if p_range[0] <= values[0] <= p_range[1]:
                p1 = values[0]
                st1 = k * p1 + b
                return True, [st1, st1 + d, p1, vtw[2]]  # 观测动作

    p_temp = p0-V**2/w
    # (3) p0-V^2/w <= p1 <= min(p0, p_max, p_max2)
    p_range = [max(p_temp, p_min), min(p0, p_max)]
    if p_range[0] <= p_range[1]:
        is_exist, values = solve_quadratic_inequality(a=k ** 2, b=2 * k * (b - ft - deltTR - ST) + 4 / w,
                                                      c=(b - ft - deltTR - ST) ** 2 - 4 / w * p0)
        if is_exist == 1:   # 恒成立
            p1 = p_range[1]
            st1 = k * p1 + b
            return True, [st1, st1 + d, p1, vtw[2]]  # 观测动作
        else:
            if p_range[0] <= values[1] <= p_range[1]:
                p1 = p_range[1]
                st1 = k * p1 + b
                return True, [st1, st1 + d, p1, vtw[2]]  # 观测动作
            if p_range[0] <= values[0] <= p_range[1]:
                p1 = values[0]
                st1 = k * p1 + b
                return True, [st1, st1 + d, p1, vtw[2]]  # 观测动作

    # (4) p1 <= p0 - V^2/w = p_temp
    p_range = [p_min, min(p_temp, p_max)]
    if p_temp >= -45:
        is_ge, value = solve_linear_inequality(a=k * V + 1, b=(-b + ft + deltTR + ST) * V + p0 + V ** 2 / w)
        if is_ge == 0:  # p1 < value
            if value >= p_range[0]:
                p1 = min(value, p_range[1])
                st1 = k * p1 + b
                return True, [st1, st1 + d, p1, vtw[2]]  # 观测动作

        elif is_ge == 1:    # p1 > value
            if value <= p_range[1]:
                p1 = p_range[1]
                st1 = k * p1 + b
                return True, [st1, st1 + d, p1, vtw[2]]  # 观测动作

    return False, []


def check_solution(solution, task_vtw, task_dur, mo, sat_arg):
    """
    :param solution: list sat_num*[task_num, 6] # 0选择的动作即任务，1所在轨道，2开始时间，3结束时间，4俯仰角，5滚动角
    :param task_vtw: array, (task_num, sat_num, max_wn, 7)
        0 卫星编号+1，1 轨道编号（从1开始），2 最早开始时间， 3 最晚开始时间，4 滚动角，5 斜率k，6 截距b
    :param task_dur:
    :param mo:  array, (task_num, sat_num, orbit_num, 2)
    :param sat_arg:
    :return:
    """
    ST = 1
    num_task, sat_num,_, _ = task_vtw.size()
    task_sat_state = np.zeros([num_task, sat_num + 1])
    for si in range(sat_num):
        sat_state = [1, 0, 0, 0, sat_arg.memory, sat_arg.energy]
        # 0轨道，1ft，2pa,3ra, 4, 5
        for ti in range(solution[si].shape[0]):
            task_id = int(solution[si][ti, 0])
            task_sat_state[task_id, si] += 1    # 累计一次，后续判断有无多算
            # 判断观测持续时间是否满足约束,存在很小的误差
            if abs(solution[si][ti, 3] - solution[si][ti, 2] - task_dur[task_id]) > 0.1:
                print('0Error-观测持续时间不满足要求:', si, ti, task_id)
            # 找出对应的窗口
            oi = int(solution[si][ti, 1])
            if not mo[task_id, si, oi, 0]:
                print('1Error-该任务在该轨道上没有窗口:', si, ti, task_id, oi)
            wi = int(mo[task_id, si, oi, 1])
            if abs(task_vtw[task_id, si, wi, 5] * solution[si][ti, 4] + task_vtw[task_id, si, wi, 6] -
                   solution[si][ti, 2]) > 1:
                print('2Error-开始时间计算误差较大:', si, ti, task_id)
            # 以上是对结果准确性的判断，下面是约束判断
            # 判断转换时间是否充足
            if transT(abs(sat_state[2]-solution[si][ti, 4]), sat_arg.AngleVelocity, sat_arg.AngleAcceleration) + \
                transT(abs(sat_state[3] - solution[si][ti, 5]), sat_arg.AngleVelocity, sat_arg.AngleAcceleration) > \
                solution[si][ti, 2] - sat_state[1]:     # 转换时间大于时间间隔
                print('3Error-转换时间大于时间间隔:', si, ti, task_id)
            if oi < sat_state[0]:
                print('4Error-轨道不正确:', si, ti, task_id, oi, sat_state[0])
            if oi > sat_state[0]:
                rest_memory = sat_arg.memory
                rest_energy= sat_arg.energy
            else:
                rest_memory = sat_state[4]
                rest_energy = sat_state[5]
            conM = task_dur[task_id] * sat_arg.mc_rate
            conE = task_dur[task_id] * sat_arg.eco_rate + (abs(sat_state[2]-solution[si][ti, 4]) +
                                                           abs(sat_state[3]-solution[si][ti, 5])) * sat_arg.ect_rate
            if conM > rest_memory or conE > rest_energy:
                print('5Error-不满足存储和能量约束:', si, ti, task_id)
            # 更新卫星状态
            sat_state[0] = oi
            sat_state[1] = solution[si][ti, 3]
            sat_state[2] = solution[si][ti, 4]
            sat_state[3] = solution[si][ti, 5]
            sat_state[4] = rest_memory-conM
            sat_state[5] = rest_energy-conE
            # 最后检查唯一性约束
    task_sat_state[:, -1] = task_sat_state[:, :sat_num].sum(axis=1)
    if (task_sat_state > 1).any():
        print('6Error-不满足唯一性约束')
    return task_sat_state



