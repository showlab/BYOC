import csv
import numpy as np


def cal_x_y(path='../landmark_results/00048.csv'):
    list_x = []
    list_y = []

    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        for row in reader:
            list_x.append(row[0])
            list_y.append(row[1])

    list_x = list_x[19:]
    list_y = list_y[19:]

    list_res = []
    for i in range(50):
        list_res.append((float(list_x[i]), float(list_y[i])))

    return list_res


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def average_euclidean_distance(points1, points2):
    if len(points1) != len(points2):
        raise ValueError("两组点的数量必须相同")

    return np.mean([euclidean_distance(p1, p2) for p1, p2 in zip(points1, points2)])


if __name__ == '__main__':
    points1 = cal_x_y('../landmark_results/00048.csv')
    points2 = cal_x_y('../landmark_results/01447.csv')

    avg_distance = average_euclidean_distance(points1, points2)
    print("平均欧氏距离:", avg_distance)
