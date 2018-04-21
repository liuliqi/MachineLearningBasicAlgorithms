from numpy import *
import numpy as np
from collections import defaultdict

class Kmeans(object):
    def __init__(self):
        pass

    # 初始化K个质心
    def initialize_centroids(self, points, k):
        array = np.array(points)
        dimensions_count = len(array[0])
        centroids_list = zeros((k, dimensions_count),dtype=float)
        # 求每一列的最大值，最小值
        for dim_index in range(dimensions_count):
            max_value_of_dim_index = max(array[:, dim_index])
            min_value_of_dim_index = min(array[:, dim_index])
            value_range_of_dim_index = float(max_value_of_dim_index - min_value_of_dim_index)
            # 对质心矩阵数组的每一列赋随机值，共K行dimensions_count列
            centroids_list[:, dim_index] = min_value_of_dim_index + value_range_of_dim_index * random.rand(k)
        return centroids_list

    # 计算两个向量之间的欧氏距离
    def calculate_point2centroids_distance(self, vector_a, vector_b):
        square_sum = 0
        for dim_index in range(len(vector_a)):
            square_sum += (vector_a[dim_index] - vector_b[dim_index])**2
        return sqrt(square_sum)

    # 将每个样本点分配到距离其最近的质心中
    def assign_point_to_centroid(self, points, centroids):
        assignment_list = []
        for point in points:
            min_distance = inf
            centriod_index_of_min_distance = 0
            for centroid_index in range(len(centroids)):
                distance = Kmeans().calculate_point2centroids_distance(point, centroids[centroid_index])
                # 更新离当前样本点最近的质心的距离，同时更新最小距离质心的索引
                if distance < min_distance:
                    min_distance = distance
                    centriod_index_of_min_distance = centroid_index
            assignment_list.append(centriod_index_of_min_distance)
        return assignment_list

    # 计算簇内的平均距离，返回一个簇的新的中心坐标
    def calculate_cluster_mean_distance(self, points):
        # 样本点的列数
        dimensions_count = len(points[0])
        new_centroid = []
        # 更新每一列的平均值
        for dim_index in range(dimensions_count):
            dim_sum = 0
            for point in points:
                dim_sum += point[dim_index]
            new_centroid.append(dim_sum / float(len(points)))
        return new_centroid

    # 更新簇中心点
    def update_centroid(self, assignment_list, points):
        new_means_dict = defaultdict(list)
        new_centroids_list = []
        for assignment, point in list(zip(assignment_list, points)):
            new_means_dict[assignment].append(point)

        # 计算并更新所有簇的簇中心 new_centroids=[[簇1x, 簇1y],[簇2x,簇2y],[簇3x,簇3y]]
        for points in new_means_dict.values():
            new_centroids_list.append(Kmeans().calculate_cluster_mean_distance(points))

        return new_centroids_list

    # k-means核心算法
    def kmeans(self, points, k):
        k_centroids = Kmeans().initialize_centroids(points, k)
        assignment_list = Kmeans().assign_point_to_centroid(points, k_centroids)
        old_assignment_list = None
        while old_assignment_list != assignment_list:
            new_centroids_list = Kmeans().update_centroid(assignment_list, points)
            old_assignment_list = assignment_list
            assignment_list = Kmeans().assign_point_to_centroid(points, new_centroids_list)
        return list(zip(assignment_list, points))


if __name__ == "__main__":
    array = [[1.658985,4.285136], [-3.453687, 3.424321], [4.838138, -1.151539], [-5.379713, -3.362104]]
    print(Kmeans().kmeans(array, 3))
