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
        self.field_of_view = 120  # 度
        self.resolution_theta = 1
        self.max_distance = 50  # 有效距离
        self.gridmap = env_map
        ray_nums = int(self.field_of_view / self.resolution_theta)
        self.angles = np.linspace(-self.field_of_view / 2, self.field_of_view / 2, ray_nums)
        self.obstacle = np.array([[0, 0]])  # 障碍物坐标
        self.map_matrix = np.zeros_like(self.gridmap.map_matrix)

    def updateState(self, state):
        self.state = state

    def updateObstacle(self):
        """
            获得检测到的障碍物,is_intersect
            更新到self.obstacle中
        """
        cur_angles = self.angles + np.ones_like(self.angles) * (self.state[2])
        obs = []
        for _, angle in enumerate(cur_angles):
            start_loc = np.array(self.state[0:2]) / self.gridmap.resolution
            x1, y1 = int(start_loc[0]), int(start_loc[1])
            ray_direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            end_state = start_loc + ray_direction * self.max_distance
            x2, y2 = int(end_state[0]), int(end_state[1])
            dy = y2 - y1
            dx = x2 - x1
            reverse = False
            if abs(dy) > abs(dx):
                x1, y1 = y1, x1
                x2, y2 = y2, x2
                reverse = True
            k = (y2 - y1) / (x2 - x1 + 1e-12)
            if x1 > x2:
                step = -1
            else:
                step = 1
            for x in range(x1, x2 + 1, step):
                y = (x - x1) * k + y1
                searchedState = np.array([y, x]) if reverse else np.array([x, y])
                if searchedState[0] < self.gridmap.borders[0] or searchedState[0] > self.gridmap.borders[1] or \
                        searchedState[1] < self.gridmap.borders[2] or searchedState[1] > self.gridmap.borders[3] or \
                        self.gridmap.isobstacle(searchedState * self.gridmap.resolution):
                    occupyed_y = int(min(max(int(searchedState[1]), self.gridmap.borders[2]), self.gridmap.borders[3]))
                    occupyed_x = int(min(max(int(searchedState[0]), self.gridmap.borders[0]), self.gridmap.borders[1]))
                    self.map_matrix[occupyed_y][occupyed_x] = 1
                    # obs.append(searchedState * self.gridmap.resolution)
                    break
            # ray_direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            # max_distance_grid = int(self.max_distance / self.gridmap.resolution)
            # for distance in range(0, max_distance_grid):
            #     ray_end = self.state[0:2] + ray_direction * distance * self.gridmap.resolution
            #     if ray_end[0] < self.gridmap.x_min or ray_end[0] > self.gridmap.x_max or \
            #             ray_end[1] < self.gridmap.y_min or ray_end[1] > self.gridmap.y_max or \
            #             self.gridmap.isobstacle(ray_end):
            #         obs.append(ray_end)
            #         break
        # if len(obs) is 0:
        #     return
        # if self.obstacle is None:
        #     self.obstacle = np.array(obs)
        # else:
        #     self.obstacle = np.concatenate((self.obstacle, np.array(obs)), axis=0)
        #     _, indices = np.unique((self.obstacle / self.gridmap.resolution).astype(np.int64), axis=0,
        #                            return_index=True)
        #     self.obstacle = self.obstacle[indices]
        cost_map = binary_dilation(self.map_matrix, np.ones((2, 2))).astype(
            self.map_matrix.dtype)
        positions = np.where(cost_map == 1)
        self.obstacle = np.array([positions[1], positions[0]]).T * self.gridmap.resolution

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
        self.borders = np.array([self.x_min, self.x_max, self.y_min, self.y_max]) / self.resolution
        self.borders.astype(np.int64)
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
        self.obstacles = np.array([positions[1], positions[0]]).T * self.resolution
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
        inflated_size = 1
        self.cost_map = binary_dilation(self.map_matrix, np.ones((inflated_size, inflated_size))).astype(
            self.map_matrix.dtype)
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
