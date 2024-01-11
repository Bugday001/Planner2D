import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from params import Params


class drawnData:
    def __init__(self):
        self.data = None

    def updateDrawnData(self, data):
        self.data = data

    def getData(self):
        return self.data


class drawVision:
    def __init__(self, envMap, params, data) -> None:
        self.dt = params.dt_
        self.mapMatrix = envMap.map_matrix
        self.resolution = envMap.resolution

        self.barName = ["X-Vel", "Y-Vel", "X-Acc", "Y-Acc"]
        self.maxVel_Acc = [params.max_vel_, params.max_vel_, params.max_acc_, params.max_acc_]

        self.fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        gs = GridSpec(1, 4, figure=self.fig)
        self.ax1 = self.fig.add_subplot(gs[0, 0:3])
        self.ax2 = self.fig.add_subplot(gs[0, -1])
        # 绘图属性
        self.ax1.set_xlim(-1, 11)
        self.ax1.set_ylim(-1, 11)
        self.ax1.imshow(self.mapMatrix, cmap='binary', extent=(0, 10, 0, 10), origin='lower')  # lower:将原点放在左下
        self.ax1.grid(color='black', linewidth=1)
        self.ax2.set_xlim(-1, 4)
        self.ax2.set_ylim(0, max(self.maxVel_Acc)+1)
        self.ax2.bar(self.barName, self.maxVel_Acc)
        # 绘图元素,
        self.polygonAx1, = self.ax1.plot([], [], color="green")
        self.globalLineAx1, = self.ax1.plot([], [], "-o")
        self.localLineAx1, = self.ax1.plot([], [], color="black")
        self.obsAx1 = self.ax1.scatter([], [], color="red", s=1)
        self.uavAx1 = self.ax1.arrow(0, 0, 0, 0, width=0.05, head_width=0.2, head_length=0.25, fc='red', ec='red')
        self.barAx2 = self.ax2.bar(self.barName, self.maxVel_Acc)
        # 使用animation绘制，耗时低
        self.getData = data.getData
        _ = animation.FuncAnimation(self.fig, self.update, interval=100, blit=True, fargs=(self.getData,))
        plt.show()

    def update(self, _, getData):
        data = getData()
        if data is not None:
            self.drawBasicInfo(*data)
        return self.polygonAx1, self.globalLineAx1, self.obsAx1, self.localLineAx1, self.uavAx1, *[i for i in self.barAx2]

    def drawBasicInfo(self, decomp_polygons, line_points, cur_state, obs_points, trajectory):
        # 安全走廊
        x = []
        y = []
        for polygon in decomp_polygons:
            verticals = polygon.getVerticals()
            # 绘制多面体顶点
            x.extend([v[0] for v in verticals] + [verticals[0][0]])
            y.extend([v[1] for v in verticals] + [verticals[0][1]])
        self.polygonAx1.set_data(x, y)
        # 全局轨迹
        self.globalLineAx1.set_data(line_points.T[0], line_points.T[1])

        # 传感器数据
        self.obsAx1 = self.ax1.scatter(obs_points.T[0], obs_points.T[1], color="red", s=1)
        # 规划轨迹
        # 轨迹
        t_samples = np.linspace(0, trajectory.time_segments_[-1], 100)
        # 位置可视化
        x_samples, y_samples = list(), list()
        for t_sample in t_samples:
            x, y = trajectory.getPos(t_sample)
            x_samples.append(x)
            y_samples.append(y)
        self.localLineAx1.set_data(x_samples, y_samples)
        # self.ax1.plot(x_samples, y_samples, color="black")
        x_samples, y_samples = list(), list()
        # for seg_t in trajectory.time_segments_:
        #     x, y = trajectory.getPos(seg_t)
        #     x_samples.append(x)
        #     y_samples.append(y)
        # self.ax1.plot(x_samples, y_samples, "o", color="red")

        # 实时状态
        x, y = cur_state[0:2]
        l = 0.25
        dx, dy = l*np.cos(np.radians(cur_state[2])), l*np.sin(np.radians(cur_state[2]))
        self.uavAx1 = self.ax1.arrow(x, y, dx, dy, width=0.05, head_width=0.2, head_length=0.25, fc='red', ec='red')
        # axs1
        vel = np.abs(cur_state[4:6])
        acc = np.abs(cur_state[6:8])
        data = [vel[0], vel[1], acc[0], acc[1]]
        for i, rect in enumerate(self.barAx2):
            rect.set_height(data[i])  # 将每个条形的高度设置为4

