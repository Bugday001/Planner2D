import numpy as np
from convex_decomp import *
import osqp
from scipy import sparse
import matplotlib.pyplot as plt
from myVision import drawVision, drawnData
from myEnv import envMap, envUAV
from trajectory_optmization import TrajectoryOptimizer, getAngle
from params import Params
import threading


class BugPlanner:
    WAIT_TASK = 0
    FIRST_PLAN = 1
    REPLAN = 2

    def __init__(self, globalMap, uav, params):
        self.state_machine_ = self.WAIT_TASK
        self.cur_state_ = None
        self.final_state_ = None
        self.goal_state_ = None
        self.globalMap_ = globalMap
        self.uav_ = uav
        self.globalPlanner_ = JPS()
        self.globalLine_ = None
        self.dt_ = params.dt_
        self.max_vel_ = params.max_vel_
        self.max_acc_ = params.max_acc_

    def resetPlanner(self):
        self.state_machine_ = self.WAIT_TASK
        self.cur_state_ = None
        self.final_state_ = None
        self.goal_state_ = None

    def getPoint(self, state):
        return Point(int(state[1] / self.globalMap_.resolution), int(state[0] / self.globalMap_.resolution))

    def setTarget(self, cur_state, final_state):
        self.cur_state_ = cur_state
        self.final_state_ = final_state
        self.state_machine_ = self.FIRST_PLAN
        self.uav_ = envUAV(self.globalMap_, self.cur_state_)
        self.localPlannerWhile()

    def globalSearch(self, start_point, goal_point):
        searched_path = self.globalPlanner_.search(start_point, goal_point, self.globalMap_.cost_map)
        if searched_path is None:
            searched_path = self.globalPlanner_.search(start_point, goal_point, self.globalMap_.map_matrix)
        if searched_path is not None:
            searched_path = self.globalPlanner_.pruning(searched_path)
            xs, ys = list(), list()
            for p in searched_path:
                xs.append(p.x_)
                ys.append(p.y_)
            line_points = np.array([ys, xs]).T * map1.resolution
        else:
            raise ValueError("JPS cannot find a way!")
        return line_points

    def getTraj(self, start_state, end_state, line_points, obs_points):
        # 进行凸分解
        convex_decomp = ConvexDecomp(2)
        decomp_polygons = convex_decomp.decomp(line_points, obs_points, False)
        # 进行轨迹优化
        traj_opt = TrajectoryOptimizer(self.max_vel_, self.max_acc_, 80)
        piece_wise_trajectory = traj_opt.optimize(start_state, end_state, line_points, decomp_polygons)
        return decomp_polygons, piece_wise_trajectory

    def localPlannerWhile(self):
        cur_point = self.getPoint(self.cur_state_)
        final_point = self.getPoint(self.final_state_)
        self.globalLine_ = self.globalSearch(cur_point, final_point)
        t = 0
        obs_points = map1.obstacles  # uav.getObstacle()
        decomp_polygons, piece_wise_trajectory = self.getTraj(self.cur_state_, self.final_state_, self.globalLine_,
                                                              obs_points)
        while (final_point - cur_point).value() > 2:
            start_time = time.time()
            x, y = piece_wise_trajectory.getPos(t)
            vx, vy = piece_wise_trajectory.getVel(t)
            ax, ay = piece_wise_trajectory.getAcc(t)
            yew = piece_wise_trajectory.getYew(t)
            dot_yew = piece_wise_trajectory.getDYew(t)
            self.cur_state_ = [x, y, yew, dot_yew, vx, vy, ax, ay]
            cur_point = self.getPoint(self.cur_state_)
            self.uav_.updateState(self.cur_state_)
            self.uav_.updateObstacle()  # 障碍物
            t += self.dt_
            data2draw.updateDrawnData((decomp_polygons, self.globalLine_, self.cur_state_, obs_points, piece_wise_trajectory))
            time.sleep(self.dt_)
            print(time.time() - start_time)
        self.resetPlanner()
    """
        def localPlanner(self):
        if self.state_machine_ is self.WAIT_TASK:
            return
        cur_state = self.getPoint(self.cur_state_)
        final_point = self.getPoint(self.final_state_)
        if self.state_machine_ is self.FIRST_PLAN:
            self.globalLine_ = self.globalSearch(cur_state, final_point)
            self.state_machine_ = self.REPLAN
        elif self.state_machine_ is self.REPLAN:
            if (final_point - cur_state).value() < 2:
                self.state_machine_ = self.WAIT_TASK
                self.resetPlanner()
                return
            elif self.goal_state_ is None:
                pass
    """


def loopTest():
    myPlanner = BugPlanner(map1, None, params)
    for i in range(10):
        # 飞行环境
        height, width = map1.map_matrix.shape
        while True:
            start_point = Point(np.random.randint(10, height - 10), np.random.randint(10, width - 10))
            if map1.cost_map[start_point.y_][start_point.x_] == 0:
                break
        while True:
            goal_point = Point(np.random.randint(10, height - 10), np.random.randint(10, width - 10))
            if map1.cost_map[goal_point.y_][goal_point.x_] == 0 and (goal_point - start_point).value() > 20:
                break
        # 初始和终止状态
        # start
        x0, y0, vx0, vy0, ax0, ay0 = start_point.x_ * map1.resolution, start_point.y_ * map1.resolution, 0.0, 0.001, 0, 0
        theta0 = getAngle([vx0, vy0])
        dot_theta0 = getAngle([ax0, ay0])
        start_state = np.array([x0, y0, theta0, dot_theta0, vx0, vy0, ax0, ay0])
        # end
        x1, y1, vx1, vy1, ax1, ay1 = goal_point.x_ * map1.resolution, goal_point.y_ * map1.resolution, 0, 0, 0, 0
        theta1 = getAngle([vx1, vy1])
        dot_theta1 = getAngle([ax1, ay1])
        end_state = np.array([x1, y1, theta1, dot_theta1, vx1, vy1, ax1, ay1])
        myPlanner.setTarget(start_state, end_state)
    print("FINISHED!")


if __name__ == "__main__":
    dt = 0.1
    max_vel, max_acc = 4, 15
    params = Params(dt, max_vel, max_acc)
    map1 = envMap()
    map1.initMapFromImg()
    map1.inflateMap()
    data2draw = drawnData()
    test_thread = threading.Thread(target=loopTest)
    test_thread.start()
    fig = drawVision(map1, params, data2draw)
    test_thread.join()
