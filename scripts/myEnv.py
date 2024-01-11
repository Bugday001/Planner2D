"""
            ^90°
            |   /60°
            |  /
            | /
180°        |/                      0°
————————————————————————————————————>
            |                       360°
            |
            |
            |270°                       
"""
from asyncio.windows_events import NULL
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.ndimage import binary_dilation


class envUAV:
    def __init__(self, env_map, state=np.array([0, 0, 0, 0, 0, 0, 0, 0])) -> None:
        self.state = state  # x,y,theta,dot_theta, vx,vy,ax,ay
        self.field_of_view = 360  # 度
        self.resolution_theta = 1
        self.max_distance = 5  # 有效距离
        self.gridmap = env_map
        ray_nums = int(self.field_of_view / self.resolution_theta)
        self.angles = np.linspace(-self.field_of_view / 2, self.field_of_view / 2, ray_nums)
        self.obstacle = NULL  # 障碍物坐标

    def updateState(self, state):
        self.state = state

    def updateObstacle(self):
        """
            获得检测到的障碍物,is_intersect
            更新到self.obstacle中
        """
        cur_angles = self.angles + np.ones_like(self.angles) * (self.state[2])
        max_distance_grid = int(self.max_distance / self.gridmap.resolution)
        obs = []
        for _, angle in enumerate(cur_angles):
            ray_direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            for distance in range(0, max_distance_grid):
                ray_end = self.state[0:2] + ray_direction * distance * self.gridmap.resolution
                if ray_end[0] < self.gridmap.x_min or ray_end[0] > self.gridmap.x_max or \
                        ray_end[1] < self.gridmap.y_min or ray_end[1] > self.gridmap.y_max or \
                        self.gridmap.isobstacle(ray_end):
                    obs.append(ray_end)
                    break
        if self.obstacle is NULL:
            self.obstacle = np.array(obs)
        else:
            self.obstacle = np.concatenate((self.obstacle, np.array(obs)), axis=0)
            _, indices = np.unique((self.obstacle / self.gridmap.resolution).astype(np.int64), axis=0, return_index=True)
            self.obstacle = self.obstacle[indices]

    def getObstacle(self):
        """
            获取积累的障碍物位置
        """
        return self.obstacle

    def drawObs(self, obs):
        # print(obs)
        plt.imshow(self.gridmap.map_matrix, cmap='binary', extent=(0, 10, 0, 10), origin='lower')
        plt.grid(color='black', linewidth=1)
        plt.scatter(obs.T[0], obs.T[1], color="red", s=1)
        plt.show()


class envMap:
    def __init__(self) -> None:
        self.x_min = 0
        self.x_max = 10
        self.y_min = 0
        self.y_max = 10
        self.resolution = 0.1
        # 创建栅格地图
        grid_x = np.arange(self.x_min, self.x_max + self.resolution, self.resolution)
        grid_y = np.arange(self.y_min, self.y_max + self.resolution, self.resolution)
        # 初始化地图矩阵
        self.map_matrix = np.zeros((len(grid_y), len(grid_x)))
        self.cost_map = self.map_matrix
        self.obstacles = np.array([])

    def initMapFromMatrix(self):
        width = 10
        heigh = 20
        loc = [[20, 80], [60, 80], [30, 40], [70, 40]]
        obs = []
        for x, y in loc:
            for c in range(width):
                for r in range(heigh):
                    self.map_matrix[int(self.y_max / self.resolution) - y + r][x + c] = 1
                    obs.append([x + c, int(self.y_max / self.resolution) - y + r])
        self.obstacles = np.array(obs)
        # np.savetxt('map_matrix.txt', self.map_matrix, fmt='%d', delimiter=' ')

    def initMapFromImg(self):
        map_path = os.path.dirname(__file__) + "/../data/test.png"
        if not os.path.exists(map_path):
            print("occupancy grid does not exist in ", map_path)
            exit(0)
        orgin_image = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        resize_img = cv2.resize(orgin_image, self.map_matrix.shape)
        _, binary_img = cv2.threshold(resize_img, 150, 255, cv2.THRESH_BINARY)
        self.map_matrix = 1 - (np.array(binary_img, dtype=np.int8) * -1)
        positions = np.where(self.map_matrix == 1)
        self.obstacles = np.array([positions[1], positions[0]]).T*self.resolution
        # plt.figure()
        # plt.imshow(self.map_matrix, cmap='binary', extent=(0, 10, 0, 10), origin='lower')
        # plt.grid(color='black', linewidth=1)
        # plt.show()
        # np.savetxt('map_matrix.txt', self.map_matrix, fmt='%d', delimiter=' ')

    def inflateMap(self):
        """
        简单的全1膨胀
        :return:
        """
        inflated_size = 5
        self.cost_map = binary_dilation(self.map_matrix, np.ones((inflated_size, inflated_size))).astype(self.map_matrix.dtype)
        # np.savetxt('map_matrix.txt', self.cost_map, fmt='%d', delimiter=' ')

    def isobstacle(self, loc):
        """
            传入实际坐标，判断该位置是否为障碍物
        """
        return self.map_matrix[int((loc[1]) / self.resolution)][int(loc[0] / self.resolution)]


if __name__ == "__main__":
    m = envMap()
    m.initMapFromImg()
    m.inflateMap()
    plt.imshow(m.map_matrix, cmap='binary', extent=(0, 10, 0, 10), origin='lower')  # lower:将原点放在左下
    # 传感器数据
    plt.scatter(m.obstacles.T[0], m.obstacles.T[1], color="red", s=1)
    plt.grid(color='black', linewidth=1)
    plt.show()
    plt.imshow(m.cost_map, cmap='binary', extent=(0, 10, 0, 10), origin='lower')  # lower:将原点放在左下
    plt.grid(color='black', linewidth=1)
    plt.show()
    # uav = envUAV(m)
    # uav.updateObstacle()
    # obs = uav.getObstacle()
    # uav.drawObs(obs)
