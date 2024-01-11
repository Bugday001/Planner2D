#! /usr/bin/env python
# ! -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import heapq
import matplotlib.pyplot as plt
import time


# 向量
class Vector:
    def __init__(self, x: int, y: int) -> None:
        self.x_ = x
        self.y_ = y

    def value(self):
        return np.sqrt(self.x_ ** 2 + self.y_ ** 2)

    def norm(self):
        if self.x_ != 0:
            self.x_ = self.x_ // np.abs(self.x_)
        if self.y_ != 0:
            self.y_ = self.y_ // np.abs(self.y_)

    def __str__(self) -> str:
        return str(self.x_) + " " + str(self.y_)


# 点
class Point:
    # 构造函数
    def __init__(self, x: int, y: int) -> None:
        self.x_ = x
        self.y_ = y

    def __add__(self, other: Vector):
        return Point(self.x_ + other.x_, self.y_ + other.y_)

    def __sub__(self, other) -> Vector:
        return Vector(self.x_ - other.x_, self.y_ - other.y_)

    def __eq__(self, other) -> bool:
        return self.x_ == other.x_ and self.y_ == other.y_

    def __str__(self) -> str:
        return str(self.x_) + " " + str(self.y_)

    def hash(self):
        return (self.x_, self.y_)


# 节点
class Node:
    def __init__(self, point: Point, cost=0.0, heuristic=0.0, parent=None) -> None:
        self.point_ = point
        self.cost_ = cost
        self.heuristic_ = heuristic
        self.f_value_ = cost + heuristic
        self.parent_ = parent
        self.dire_ = Vector(0, 0)
        if self.parent_ != None:
            self.dire_ = self.point_ - parent.point_
            self.dire_.norm()

    def __lt__(self, other):
        return self.f_value_ < other.f_value_

    def hash(self):
        return self.point_.hash()


# jump point search 算法
class JPS:
    # 构造函数
    def __init__(self) -> None:
        self.height_ = None
        self.width_ = None
        self.occupancy_grid_ = None
        self.start_point_ = None
        self.goal_point_ = None
        self.candidate_dires_ = [Vector(1, 0), Vector(0, 1), Vector(0, -1), Vector(-1, 0), Vector(1, 1), Vector(1, -1),
                                 Vector(-1, 1), Vector(-1, -1)]

    # 进行路径搜索
    def search(self, start_point: Point, goal_point: Point, occupancy_grid: np.array) -> list:
        # 初始化变量
        self.height_, self.width_ = occupancy_grid.shape
        self.occupancy_grid_ = occupancy_grid
        self.start_point_ = start_point
        self.goal_point_ = goal_point
        # 构建初始节点和目标节点
        start_node = Node(start_point, heuristic=(goal_point - start_point).value())
        goal_node = Node(goal_point)
        # 构建开集合和闭集合
        open_set = [start_node]
        heapq.heapify(open_set)
        close_set = set()
        # 开始进行路径搜索
        while open_set:
            # 得到当前节点
            cur_node = heapq.heappop(open_set)
            # 判断当前节点是否在close集合中
            if cur_node.hash() in close_set:
                continue
            # 当前节点加入close集合
            close_set.add(cur_node.hash())
            # 判断当前节点是否为终点
            if cur_node.point_ == self.goal_point_:
                goal_node = cur_node
                break
            # 根据当前节点获得跳点搜索方向
            search_dires = self.getSearchDires(cur_node)
            # 根据方向进行跳点搜索
            for search_dire in search_dires:
                jump_point = self.jumpPointSearch(cur_node.point_, search_dire)
                # 进行节点构造
                if jump_point is not None:
                    # 判断跳点是否已经被搜索过
                    if jump_point.hash() in close_set:
                        continue
                    # 构建新节点
                    new_node = Node(point=jump_point, cost=cur_node.cost_ + (jump_point - cur_node.point_).value(),
                                    heuristic=(self.goal_point_ - jump_point).value(), parent=cur_node)
                    # 加入open集合
                    heapq.heappush(open_set, new_node)
        # 判断是否搜索成功
        if goal_node.parent_ is None:
            print("path search failed")
            return None
        # 进行路径回溯
        print("path search success")
        path = list()
        cur_node = goal_node
        while cur_node is not None:
            path.append(cur_node.point_)
            cur_node = cur_node.parent_
        path = list(reversed(path))
        return path

    # 获得搜索方向
    def getSearchDires(self, node: Node) -> list():
        search_dires = list()
        # 判断是否为初始节点
        if node.parent_ is None:
            # 是初始节点，全部方向待选
            for candidate_dir in self.candidate_dires_:
                # 判断待选方向是否可行
                if self.isAvailable(node.point_ + candidate_dir):
                    search_dires.append(candidate_dir)
        else:
            # 不是初始节点，判断当前节点方向是否可行
            if self.isAvailable(node.point_ + node.dire_):
                search_dires.append(node.dire_)
            # 判断当前节点方向是否为斜向
            if node.dire_.x_ != 0 and node.dire_.y_ != 0:
                # 当前方向为斜向
                # 判断当前方向在垂直方向上的投影方向是否可行
                if self.isAvailable(node.point_.x_, node.point_.y_ + node.dire_.y_):
                    search_dires.append(Vector(0, node.dire_.y_))
                # 判断当前方向在水平方向上的投影方向是否可行
                if self.isAvailable(node.point_.x_ + node.dire_.x_, node.point_.y_):
                    search_dires.append(Vector(node.dire_.x_, 0))
                # 判断当前方向斜向搜索方向是否需要搜索
                if not self.isAvailable(node.point_.x_ - node.dire_.x_, node.point_.y_) and self.isAvailable(
                        node.point_.x_, node.point_.y_ + node.dire_.y_):
                    search_dires.append(Vector(-node.dire_.x_, node.dire_.y_))
                # 判断当前方向斜向搜索方向是否需要搜索
                if not self.isAvailable(node.point_.x_, node.point_.y_ - node.dire_.y_) and self.isAvailable(
                        node.point_.x_ + node.dire_.x_, node.point_.y_):
                    search_dires.append(Vector(node.dire_.x_, -node.dire_.y_))
            else:
                # 当前方向非斜向
                if node.dire_.x_ == 0:
                    # 当前方向为垂直方向
                    # 判断右侧是否可行
                    if not self.isAvailable(node.point_ + Vector(1, 0)):
                        # 右侧不可行
                        search_dires.append(Vector(1, node.dire_.y_))
                    # 判断左侧是否可行
                    if not self.isAvailable(node.point_ + Vector(-1, 0)):
                        # 左侧不可行
                        search_dires.append(Vector(-1, node.dire_.y_))
                else:
                    # 当前方向为水平方向
                    # 判断下方是否可行
                    if not self.isAvailable(node.point_ + Vector(0, 1)):
                        # 下方不可行
                        search_dires.append(Vector(node.dire_.x_, 1))
                    # 判断上方是否可行
                    if not self.isAvailable(node.point_ + Vector(0, -1)):
                        # 上方不可行
                        search_dires.append(Vector(node.dire_.x_, -1))
        return search_dires

    # 判断是否为跳点
    def jumpPointCheck(self, cur_point: Point, direction: Vector) -> bool:
        if cur_point == self.goal_point_:
            return True
        # 判断当前点是否可行
        if not self.isAvailable(cur_point):
            return False
        # 判断当前朝向是否为斜向
        if direction.x_ != 0 and direction.y_ != 0:
            # 进行跳点判断
            # 左下能走且左不能走，或右上能走且上不能走
            if (self.isAvailable(cur_point.x_ - direction.x_, cur_point.y_ + direction.y_) and not self.isAvailable(
                    cur_point.x_ - direction.x_, cur_point.y_)) or (
                    self.isAvailable(cur_point.x_ + direction.x_, cur_point.y_ - direction.y_) and not self.isAvailable(
                cur_point.x_, cur_point.y_ - direction.y_)):
                return True
        else:
            # 进行跳点判断
            if direction.x_ != 0:
                # 水平方向
                # 右下能走且下不能走， 或右上能走且上不能走
                '''
                * 1 0 
                0 → 0
                * 1 0
                
                '''
                if (self.isAvailable(cur_point.x_ + direction.x_, cur_point.y_ + 1) and not self.isAvailable(
                        cur_point.x_, cur_point.y_ + 1)) or (
                        self.isAvailable(cur_point.x_ + direction.x_, cur_point.y_ - 1) and not self.isAvailable(
                    cur_point.x_, cur_point.y_ - 1)):
                    return True
            else:
                # 垂直方向
                '''
                0 0 0
                1 ↓ 1
                0 0 0
                                
                '''
                if (self.isAvailable(cur_point.x_ + 1, cur_point.y_ + direction.y_) and not self.isAvailable(
                        cur_point.x_ + 1, cur_point.y_)) or (
                        self.isAvailable(cur_point.x_ - 1, cur_point.y_ + direction.y_) and not self.isAvailable(
                    cur_point.x_ - 1, cur_point.y_)):
                    return True
        return False

    # 给定位置和方向，进行跳点搜索
    def jumpPointSearch(self, pre_point: Point, direction: Vector) -> Point:
        cur_point = pre_point + direction
        # 判断当前搜索方向是否为斜向
        if direction.x_ == 0 or direction.y_ == 0:
            # 不是斜向，沿着当前方向进行跳点搜索
            while self.isAvailable(cur_point):
                # 判断当前点
                if self.jumpPointCheck(cur_point, direction):
                    return cur_point
                cur_point += direction
        else:
            # 是斜向，向横纵向进行拓展，判断是否存在跳点
            horizon_direction = Vector(direction.x_, 0)
            vertical_direction = Vector(0, direction.y_)
            while self.isAvailable(cur_point):
                # 判断当前点
                if self.jumpPointCheck(cur_point, direction):
                    return cur_point
                # 判断横向
                horizon_point = Point(cur_point.x_, cur_point.y_) + horizon_direction
                while self.isAvailable(horizon_point):
                    if self.jumpPointCheck(horizon_point, horizon_direction):
                        return cur_point
                    horizon_point += horizon_direction
                # 判断纵向
                vertical_point = Point(cur_point.x_, cur_point.y_) + vertical_direction
                while self.isAvailable(vertical_point):
                    if self.jumpPointCheck(vertical_point, vertical_direction):
                        return cur_point
                    vertical_point += vertical_direction
                # 判断是否需要沿着当前方向继续进行搜索
                if not (self.isAvailable(cur_point + horizon_direction) and self.isAvailable(
                        cur_point + vertical_direction)):
                    break
                cur_point += direction
        return None

    # 判断点是否可行
    def isAvailable(self, *args) -> bool:
        if len(args) == 1:
            point = args[0]
            if 0 <= point.x_ < self.height_ and 0 <= point.y_ < self.width_ and self.occupancy_grid_[point.x_][
                point.y_] == 0:
                return True
            else:
                return False
        else:
            x, y = args[0], args[1]
            if 0 <= x < self.height_ and 0 <= y < self.width_ and self.occupancy_grid_[x][y] == 0:
                return True
            else:
                return False

    def isVisual(self, point1, point2):
        x1, y1 = point1.x_, point1.y_
        x2, y2 = point2.x_, point2.y_
        dy = y2 - y1
        dx = x2 - x1
        reverse = False
        if abs(dy) > abs(dx):
            x1, y1 = y1, x1
            x2, y2 = y2, x2
            reverse = True
        k = (y2-y1) / (x2-x1 + 1e-12)
        if x1 > x2:
            step = -1
        else:
            step = 1
        for x in range(x1, x2 + 1, step):
            y = (x - x1) * k + y1
            if (reverse and self.occupancy_grid_[int(y)][x]) or (not reverse and self.occupancy_grid_[x][int(y)]):
                return False
        return True

    def pruning(self, path):
        nums = len(path)
        if nums < 3:
            return path
        head = Node(path[nums - 1])
        tmp = head
        i = nums - 2
        while i >= 0:
            tmp.parent_ = Node(path[i])
            tmp = tmp.parent_
            i -= 1
        tmp_head = head
        tmp = tmp_head
        while 1:
            if tmp is None or tmp_head is None:
                break
            # if tmp is not None and tmp.parent_ is not None:
            #     tmp = tmp.parent_
            # else:
            #     break
            visualNode = None
            while tmp is not None and self.isVisual(tmp_head.point_, tmp.point_):
                visualNode = tmp
                tmp = tmp.parent_
            if visualNode is not None:
                tmp_head.parent_ = visualNode
                tmp_head = visualNode
            else:
                tmp_head = tmp_head.parent_
        tmp_head = head
        pruned_path = []
        while tmp_head is not None:
            pruned_path.append(tmp_head.point_)
            tmp_head = tmp_head.parent_
        return reversed(pruned_path)


if __name__ == "__main__":
    from myEnv import envMap
    # 飞行环境
    map1 = envMap()
    map1.initMapFromImg()
    map1.inflateMap()
    height, width = map1.map_matrix.shape
    while True:
        start_point = Point(np.random.randint(10, height - 10), np.random.randint(10, width - 10))
        if map1.cost_map[start_point.x_][start_point.y_] == 0:
            break
    while True:
        goal_point = Point(np.random.randint(10, height - 10), np.random.randint(10, width - 10))
        if map1.cost_map[goal_point.x_][goal_point.y_] == 0 and (goal_point - start_point).value() > 40:
            break
    start_point = Point(55, 50)
    goal_point = Point(13, 38)
    jspMap = map1.cost_map
    jps = JPS()
    searched_path = jps.search(start_point, goal_point, jspMap)
    if searched_path is not None:
        xs, ys = list(), list()
        for p in searched_path:
            xs.append(p.x_*map1.resolution)
            ys.append(p.y_*map1.resolution)
        plt.figure()
        plt.imshow(jspMap, cmap='binary', extent=(0, 10, 0, 10), origin='lower')  # lower:将原点放在左下
        plt.scatter(start_point.y_*map1.resolution, start_point.x_*map1.resolution, marker="o", color="red", s=105)
        plt.scatter(goal_point.y_*map1.resolution, goal_point.x_*map1.resolution, marker="o", color="green", s=105)
        plt.plot(ys, xs, "-o")
        plt.show()
    searched_path = jps.pruning(searched_path)
    if searched_path is not None:
        xs, ys = list(), list()
        for p in searched_path:
            xs.append(p.x_*map1.resolution)
            ys.append(p.y_*map1.resolution)
        plt.figure()
        plt.imshow(jspMap, cmap='binary', extent=(0, 10, 0, 10), origin='lower')  # lower:将原点放在左下
        plt.scatter(start_point.y_*map1.resolution, start_point.x_*map1.resolution, marker="o", color="red", s=105)
        plt.scatter(goal_point.y_*map1.resolution, goal_point.x_*map1.resolution, marker="o", color="green", s=105)
        plt.plot(ys, xs, "-o")
        plt.show()
    else:
        raise ValueError("JPS cannot find a way!")
    print(jps.isVisual(Point(52, 53), Point(55, 50)))
