#! /usr/bin/env python
# ! -*- coding: utf-8 -*-

from tracemalloc import start
import numpy as np
from convex_decomp import *
import osqp
from scipy import sparse
import matplotlib.pyplot as plt
from myVision import drawVision
from myEnv import envMap, envUAV


# 五阶伯恩斯坦多项式
class QuinticBernsteinPolynomial:
    def __init__(self, params, time_allocation):
        """
            : list
            : float
        """
        assert (len(params) == 6)
        self.params_ = params
        self.time_allocation_ = time_allocation

    # 计算值
    def value(self, t):
        u = t / self.time_allocation_
        return self.params_[0] * (1 - u) ** 5 + self.params_[1] * 5 * (1 - u) ** 4 * u + self.params_[2] * 10 * (
                1 - u) ** 3 * u ** 2 + self.params_[3] * 10 * (1 - u) ** 2 * u ** 3 + self.params_[4] * 5 * (
                       1 - u) * u ** 4 + self.params_[5] * u ** 5

    # 计算一阶导数
    def derivative(self, t):
        u = t / self.time_allocation_
        return 1 / self.time_allocation_ * (self.params_[0] * (-5) * (1 - u) ** 4 + self.params_[1] * (
                5 * (1 - u) ** 4 - 20 * (1 - u) ** 3 * u) + self.params_[2] * (
                                                    20 * (1 - u) ** 3 * u - 30 * (1 - u) ** 2 * u ** 2) +
                                            self.params_[3] * (30 * (1 - u) ** 2 * u ** 2 - 20 * (1 - u) * u ** 3) +
                                            self.params_[4] * (20 * (1 - u) * u ** 3 - 5 * u ** 4) + self.params_[
                                                5] * 5 * u ** 4)

    # 计算二阶导数
    def secondOrderDerivative(self, t):
        u = t / self.time_allocation_
        return (1 / self.time_allocation_) ** 2 * (self.params_[0] * 20 * (1 - u) ** 3 + self.params_[1] * 5 * (
                -8 * (1 - u) ** 3 + 12 * (1 - u) ** 2 * u) + self.params_[2] * 10 * (
                                                           2 * (1 - u) ** 3 - 12 * (1 - u) ** 2 * u + 6 * (
                                                           1 - u) * u ** 2) + self.params_[3] * 10 * (
                                                           6 * (1 - u) ** 2 * u - 12 * (
                                                           1 - u) * u ** 2 + 2 * u ** 3) + self.params_[
                                                       4] * 5 * (12 * (1 - u) * u ** 2 - 8 * u ** 3) + self.params_[
                                                       5] * 20 * u ** 3)

    # 计算三阶导数
    def thirdOrderDerivative(self, t):
        u = t / self.time_allocation_
        return (1 / self.time_allocation_) ** 3 * (self.params_[0] * (-60) * (1 - u) ** 2 + self.params_[1] * 5 * (
                36 * (1 - u) ** 2 - 24 * (1 - u) * u) + self.params_[2] * 10 * (
                                                           -18 * (1 - u) ** 2 + 36 * (1 - u) * u - 6 * u ** 2) +
                                                   self.params_[3] * 10 * (
                                                           6 * (1 - u) ** 2 - 36 * (1 - u) * u + 18 * u ** 2) +
                                                   self.params_[4] * 5 * (24 * (1 - u) * u - 36 * u ** 2) +
                                                   self.params_[5] * 60 * u ** 2)


# 分段轨迹
class PieceWiseTrajectory:
    def __init__(self, x_params: list, y_params: list, time_allocations: list) -> None:
        self.segment_num_ = len(time_allocations)
        self.time_segments_ = np.cumsum(time_allocations)
        self.trajectory_segments_ = list()
        for i in range(self.segment_num_):
            self.trajectory_segments_.append((QuinticBernsteinPolynomial(x_params[i], time_allocations[i]),
                                              QuinticBernsteinPolynomial(y_params[i], time_allocations[i])))

    # 根据时间获取下标
    def index(self, t):
        for i in range(self.segment_num_):
            if t <= self.time_segments_[i]:
                return i
        return None

    # 得到坐标
    def getPos(self, t):
        index = self.index(t)
        if index > 0:
            t = t - self.time_segments_[index - 1]
        return self.trajectory_segments_[index][0].value(t), self.trajectory_segments_[index][1].value(t)

    # 得到速度
    def getVel(self, t):
        index = self.index(t)
        if index > 0:
            t = t - self.time_segments_[index - 1]
        return self.trajectory_segments_[index][0].derivative(t), self.trajectory_segments_[index][1].derivative(t)

    # 得到航向
    def getYew(self, t):
        vx, vy = self.getVel(t)
        vx += 1e-12
        degree = np.degrees(np.arctan(vy / vx))
        if vx < 0:
            degree += 180
        return degree

    # 角加速度
    def getDYew(self, t):
        ax, ay = self.getAcc(t)
        ax += 1e-12
        beta = np.degrees(np.arctan(ay / ax))
        if ax < 0:
            beta += 180
        return beta

    # 得到加速度
    def getAcc(self, t):
        index = self.index(t)
        if index > 0:
            t = t - self.time_segments_[index - 1]
        return self.trajectory_segments_[index][0].secondOrderDerivative(t), self.trajectory_segments_[index][
            1].secondOrderDerivative(t)

    # 得到jerk
    def getJerk(self, t):
        index = self.index(t)
        if index > 0:
            t = t - self.time_segments_[index - 1]
        return self.trajectory_segments_[index][0].thirdOrderDerivative(t), self.trajectory_segments_[index][
            1].thirdOrderDerivative(t)


# 轨迹优化器
class TrajectoryOptimizer:
    def __init__(self, vel_max, acc_max, jerk_max) -> None:
        # 运动上限
        self.vel_max_ = vel_max
        self.acc_max_ = acc_max
        self.jerk_max_ = jerk_max
        # 得到维度
        self.dim_ = 2
        # 得到曲线阶数
        self.degree_ = 5
        # 自由度
        self.freedom_ = self.degree_ + 1

    # 进行优化
    def optimize(self, start_state, end_state, line_points, polygons):
        """
            : np.array
            : np.array
            : list[np.array]
            : list[Polygon]
        """
        assert (len(line_points) == len(polygons) + 1)
        # 得到分段数量
        segment_num = len(polygons)
        assert (segment_num >= 1)
        # 计算初始时间分配
        time_allocations = list()
        for i in range(segment_num):
            if i <= 0 or i >= segment_num - 1:
                r = 1
            else:
                r = 1
            time_allocations.append(np.linalg.norm(line_points[i + 1] - line_points[i]) / self.vel_max_ * r)
        # 进行优化迭代
        max_inter = 10
        cur_iter = 0
        while cur_iter < max_inter:
            # 进行轨迹优化
            piece_wise_trajectory = self.optimizeIter(start_state, end_state, polygons, time_allocations, segment_num)
            # 对优化轨迹进行时间调整，以保证轨迹满足运动上限约束
            cur_iter += 1
            # 计算每一段轨迹的最大速度，最大加速度，最大jerk
            condition_fit = True
            for n in range(segment_num):
                # 得到最大速度，最大加速度，最大jerk
                t_samples = np.linspace(0, time_allocations[n], 100)
                v_max, a_max, j_max = self.vel_max_, self.acc_max_, self.jerk_max_
                for t_sample in t_samples:
                    v_max = max(v_max, np.abs(piece_wise_trajectory.trajectory_segments_[n][0].derivative(t_sample)),
                                np.abs(piece_wise_trajectory.trajectory_segments_[n][1].derivative(t_sample)))
                    a_max = max(a_max, np.abs(
                        piece_wise_trajectory.trajectory_segments_[n][0].secondOrderDerivative(t_sample)), np.abs(
                        piece_wise_trajectory.trajectory_segments_[n][1].secondOrderDerivative(t_sample)))
                    j_max = max(j_max,
                                np.abs(piece_wise_trajectory.trajectory_segments_[n][0].thirdOrderDerivative(t_sample)),
                                np.abs(piece_wise_trajectory.trajectory_segments_[n][1].thirdOrderDerivative(t_sample)))
                # 判断是否满足约束条件
                if Compare.large(v_max, self.vel_max_) or Compare.large(a_max, self.acc_max_) or Compare.large(j_max,
                                                                                                               self.jerk_max_):
                    ratio = max(1, v_max / self.vel_max_, (a_max / self.acc_max_) ** 0.5,
                                (j_max / self.jerk_max_) ** (1 / 3))
                    time_allocations[n] = ratio * time_allocations[n]
                    condition_fit = False
                    # print(ratio, v_max, self.vel_max_)
            if condition_fit:
                break
        return piece_wise_trajectory

    def getQM(self, n_seg, n_order, ts):
        d_order = (n_order + 1) // 2
        Q = np.empty((0, n_order + 1, n_order + 1))
        M = np.empty((0, n_order + 1, n_order + 1))

        for k in range(1, n_seg + 1):
            # STEP 2.1: 计算第k段的Q_k
            Q_k = np.zeros((n_order + 1, n_order + 1))
            for i in range(d_order, n_order + 1):
                for j in range(d_order, n_order + 1):
                    Q_k[i, j] = (
                            np.math.factorial(i) / np.math.factorial(i - d_order) *
                            np.math.factorial(j) / np.math.factorial(j - d_order) /
                            (i + j - n_order) * ts[k - 1] ** (i + j - n_order)
                    )
            M_k = np.array([[1, 0, 0, 0, 0, 0],
                            [-5, 5, 0, 0, 0, 0],
                            [10, -20, 10, 0, 0, 0],
                            [-10, 30, -30, 10, 0, 0],
                            [5, -20, 30, -20, 5, 0],
                            [-1, 5, -10, 10, -5, 1]])  # 请实现getM函数
            if Q.shape[0] == 0:
                Q = Q_k
                M = M_k
            else:
                Q = np.block([[Q, np.zeros((Q.shape[0], Q_k.shape[1]))],
                              [np.zeros((Q_k.shape[0], Q.shape[1])), Q_k]])

                M = np.block([[M, np.zeros((M.shape[0], M_k.shape[1]))],
                              [np.zeros((M_k.shape[0], M.shape[1])), M_k]])
        return Q, M

    # 优化迭代
    def optimizeIter(self, start_state, end_state, polygons, time_allocations, segment_num):
        """
            : np.array
            : np.array
            : list[Polygon]
            : list
        """
        # 构建目标函数 inter (jerk)^2，M'*Q*M。
        # inte_jerk_square = np.array([
        #     [720.0, -1800.0, 1200.0, 0.0, 0.0, -120.0],
        #     [-1800.0, 4800.0, -3600.0, 0.0, 600.0, 0.0],
        #     [1200.0, -3600.0, 3600.0, -1200.0, 0.0, 0.0],
        #     [0.0, 0.0, -1200.0, 3600.0, -3600.0, 1200.0],
        #     [0.0, 600.0, 0.0, -3600.0, 4800.0, -1800.0],
        #     [-120.0, 0.0, 0.0, 1200.0, -1800.0, 720.0]
        # ])
        # 二次项系数
        # P = np.zeros((self.dim_ * segment_num * self.freedom_, self.dim_ * segment_num * self.freedom_))
        # for sigma in range(self.dim_):
        #     for n in range(segment_num):
        #         for i in range(self.freedom_):
        #             for j in range(self.freedom_):
        #                 index_i = sigma * segment_num * self.freedom_ + n * self.freedom_ + i
        #                 index_j = sigma * segment_num * self.freedom_ + n * self.freedom_ + j
        #                 P[index_i][index_j] = inte_jerk_square[i][j] / (time_allocations[n] ** 5)
        # P = P * 2
        # P = sparse.csc_matrix(P)

        # 求二次系数
        pQ, pM = self.getQM(segment_num, 5, time_allocations)
        P = pM.T @ pQ @ pM
        P = np.block([[P, np.zeros((P.shape[0], P.shape[1]))],
                      [np.zeros((P.shape[0], P.shape[1])), P]])
        P = sparse.csc_matrix(P)

        # 一次项系数
        q = np.zeros((self.dim_ * segment_num * self.freedom_,))

        # 构建约束条件，10个起点终点约束，还有3 * (segment_num - 1) * self.dim_个连接处的位置、速度、加速度约束
        equality_constraints_num = 5 * self.dim_ + 3 * (segment_num - 1) * self.dim_
        param_num = self.dim_ * segment_num * self.freedom_
        inequality_constraints_num = 0
        # 位置的超片面不等式约束
        for polygon in polygons:
            inequality_constraints_num += self.freedom_ * len(polygon.hyper_planes_)
        # # 速度的超片面不等式约束
        # inequality_constraints_num += (self.freedom_ - 1) * segment_num * self.dim_
        # # 加速度的超片面不等式约束
        # inequality_constraints_num += (self.freedom_ - 2) * segment_num * self.dim_
        A = np.zeros((equality_constraints_num + inequality_constraints_num, param_num))
        lb = -float("inf") * np.ones((equality_constraints_num + inequality_constraints_num,))
        ub = float("inf") * np.ones((equality_constraints_num + inequality_constraints_num,))

        # 构建等式约束条件（起点位置、速度、加速度；终点位置、速度；连接处的零、一、二阶导数）
        # 起点x位置
        A[0][0] = 1
        lb[0] = start_state[0]
        ub[0] = start_state[0]
        # 起点y位置
        A[1][segment_num * self.freedom_] = 1
        lb[1] = start_state[1]
        ub[1] = start_state[1]
        # 起点x速度
        A[2][0] = -5 / time_allocations[0]
        A[2][1] = 5 / time_allocations[0]
        lb[2] = start_state[4]
        ub[2] = start_state[4]
        # 起点y速度
        A[3][segment_num * self.freedom_] = -5 / time_allocations[0]
        A[3][segment_num * self.freedom_ + 1] = 5 / time_allocations[0]
        lb[3] = start_state[5]
        ub[3] = start_state[5]
        # 起点x加速度
        A[4][0] = 20 / time_allocations[0] ** 2
        A[4][1] = -40 / time_allocations[0] ** 2
        A[4][2] = 20 / time_allocations[0] ** 2
        lb[4] = start_state[6]
        ub[4] = start_state[6]
        # 起点y加速度
        A[5][segment_num * self.freedom_] = 20 / time_allocations[0] ** 2
        A[5][segment_num * self.freedom_ + 1] = -40 / time_allocations[0] ** 2
        A[5][segment_num * self.freedom_ + 2] = 20 / time_allocations[0] ** 2
        lb[5] = start_state[7]
        ub[5] = start_state[7]
        # 终点x位置
        A[6][segment_num * self.freedom_ - 1] = 1
        lb[6] = end_state[0]
        ub[6] = end_state[0]
        # 终点y位置
        A[7][self.dim_ * segment_num * self.freedom_ - 1] = 1
        lb[7] = end_state[1]
        ub[7] = end_state[1]
        # 终点x速度
        A[8][segment_num * self.freedom_ - 1] = 5 / time_allocations[-1]
        A[8][segment_num * self.freedom_ - 2] = -5 / time_allocations[-1]
        lb[8] = end_state[4]
        ub[8] = end_state[4]
        # 终点y速度
        A[9][self.dim_ * segment_num * self.freedom_ - 1] = 5 / time_allocations[-1]
        A[9][self.dim_ * segment_num * self.freedom_ - 2] = -5 / time_allocations[-1]
        lb[9] = end_state[5]
        ub[9] = end_state[5]

        # 连接处的零阶导数相等
        constraints_index = 10
        for sigma in range(self.dim_):
            for n in range(segment_num - 1):
                A[constraints_index][sigma * segment_num * self.freedom_ + n * self.freedom_ + self.freedom_ - 1] = 1
                A[constraints_index][sigma * segment_num * self.freedom_ + (n + 1) * self.freedom_] = -1
                lb[constraints_index] = 0
                ub[constraints_index] = 0
                constraints_index += 1
        # 连接处的一阶导数相等
        for sigma in range(self.dim_):
            for n in range(segment_num - 1):
                A[constraints_index][sigma * segment_num * self.freedom_ + n * self.freedom_ + self.freedom_ - 1] = 5 / \
                                                                                                                    time_allocations[
                                                                                                                        n]
                A[constraints_index][sigma * segment_num * self.freedom_ + n * self.freedom_ + self.freedom_ - 2] = -5 / \
                                                                                                                    time_allocations[
                                                                                                                        n]
                A[constraints_index][sigma * segment_num * self.freedom_ + (n + 1) * self.freedom_] = 5 / \
                                                                                                      time_allocations[
                                                                                                          n + 1]
                A[constraints_index][sigma * segment_num * self.freedom_ + (n + 1) * self.freedom_ + 1] = -5 / \
                                                                                                          time_allocations[
                                                                                                              n + 1]
                lb[constraints_index] = 0
                ub[constraints_index] = 0
                constraints_index += 1
        # 连接处的二阶导数相等
        for sigma in range(self.dim_):
            for n in range(segment_num - 1):
                A[constraints_index][sigma * segment_num * self.freedom_ + n * self.freedom_ + self.freedom_ - 1] = 20 / \
                                                                                                                    time_allocations[
                                                                                                                        n] ** 2
                A[constraints_index][
                    sigma * segment_num * self.freedom_ + n * self.freedom_ + self.freedom_ - 2] = -40 / \
                                                                                                   time_allocations[
                                                                                                       n] ** 2
                A[constraints_index][sigma * segment_num * self.freedom_ + n * self.freedom_ + self.freedom_ - 3] = 20 / \
                                                                                                                    time_allocations[
                                                                                                                        n] ** 2
                A[constraints_index][sigma * segment_num * self.freedom_ + (n + 1) * self.freedom_] = -20 / \
                                                                                                      time_allocations[
                                                                                                          n + 1] ** 2
                A[constraints_index][sigma * segment_num * self.freedom_ + (n + 1) * self.freedom_ + 1] = 40 / \
                                                                                                          time_allocations[
                                                                                                              n + 1] ** 2
                A[constraints_index][sigma * segment_num * self.freedom_ + (n + 1) * self.freedom_ + 2] = -20 / \
                                                                                                          time_allocations[
                                                                                                              n + 1] ** 2
                lb[constraints_index] = 0
                ub[constraints_index] = 0
                constraints_index += 1

        # 构建不等式约束条件
        # n_: 法向量, d_: 平移向量
        for n in range(segment_num):
            for k in range(self.freedom_):
                for hyper_plane in polygons[n].hyper_planes_:
                    A[constraints_index][n * self.freedom_ + k] = hyper_plane.n_[0]
                    A[constraints_index][segment_num * self.freedom_ + n * self.freedom_ + k] = hyper_plane.n_[1]
                    ub[constraints_index] = np.dot(hyper_plane.n_, hyper_plane.d_)
                    constraints_index += 1

        # 速度约束
        # for n in range(segment_num):
        #     for k in range(self.degree_):
        #         A[constraints_index][n * self.freedom_ + k] = -self.degree_
        #         A[constraints_index][n * self.freedom_ + k + 1] = self.degree_
        #         ub[constraints_index] = self.vel_max_
        #         lb[constraints_index] = -self.vel_max_
        #         constraints_index += 1
        # for n in range(segment_num):
        #     for k in range(self.degree_):
        #         A[constraints_index][segment_num * self.freedom_ + n * self.freedom_ + k] = -self.degree_
        #         A[constraints_index][segment_num * self.freedom_ + n * self.freedom_ + k + 1] = self.degree_
        #         ub[constraints_index] = self.vel_max_
        #         lb[constraints_index] = -self.vel_max_
        #         constraints_index += 1
        # # 加速度约束
        # pAcc = self.degree_*(self.degree_-1)
        # for n in range(segment_num):
        #     for k in range(self.degree_-1):
        #         A[constraints_index][n * self.freedom_ + k] = pAcc
        #         A[constraints_index][n * self.freedom_ + k + 1] = -2*pAcc
        #         A[constraints_index][n * self.freedom_ + k + 2] = pAcc
        #         ub[constraints_index] = self.acc_max_
        #         lb[constraints_index] = -self.acc_max_
        #         constraints_index += 1
        # for n in range(segment_num):
        #     for k in range(self.degree_-1):
        #         A[constraints_index][segment_num * self.freedom_ + n * self.freedom_ + k] = pAcc
        #         A[constraints_index][segment_num * self.freedom_ + n * self.freedom_ + k + 1] = -2*pAcc
        #         A[constraints_index][segment_num * self.freedom_ + n * self.freedom_ + k + 2] = pAcc
        #         ub[constraints_index] = self.acc_max_
        #         lb[constraints_index] = -self.acc_max_
        #         constraints_index += 1
        # assert (constraints_index == equality_constraints_num + inequality_constraints_num)
        A = sparse.csc_matrix(A)

        # 进行qp求解
        prob = osqp.OSQP()
        # minimize 0.5 x' P x + q' x subject to l <= A x <= u
        prob.setup(P, q, A, lb, ub, warm_start=True, verbose=False)
        res = prob.solve()
        if res.info.status != "solved":
            print(start_state, end_state)
            raise ValueError("OSQP did not solve the problem!")

        # 根据参数进行轨迹解析
        trajectory_x_params, trajectory_y_params = list(), list()
        for n in range(segment_num):
            trajectory_x_params.append(res.x[self.freedom_ * n: self.freedom_ * (n + 1)])
            trajectory_y_params.append(res.x[
                                       segment_num * self.freedom_ + self.freedom_ * n: segment_num * self.freedom_ + self.freedom_ * (
                                               n + 1)])
        piece_wise_trajectory = PieceWiseTrajectory(trajectory_x_params, trajectory_y_params, time_allocations)

        return piece_wise_trajectory


def getAngle(data):
    """
    
    """
    theta = np.degrees(np.arctan(data[1] / (data[0] + 1e-12)))
    if data[0] < 0:
        theta += 180
    return theta


if __name__ == "__main__":
    pass
