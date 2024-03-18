import os
import random

import numpy as np
import json
import csv
from tqdm import tqdm

conflict_pairs = [
    # Artificial filter
    [1, 3],
    [2, 3],
    [6, 13],
    [7, 13],
    [9, 17],
    [12, 17],
    [17, 18],
    [16, 11],
    [16, 15],
    [22, 13],
    [6, 23],
    [7, 23],
    [22, 23],
    # name-based filter
    [12, 13],
    [14, 15],
    [9, 18],

]


def constraint(weight):
    for conf in conflict_pairs:
        if weight[conf[0] - 1] >= weight[conf[1] - 1]:
            weight[conf[1] - 1] = 0
        else:
            weight[conf[0] - 1] = 0

    return weight


def generate():
    seed = 123
    np.random.seed(seed)
    num_ids = 10
    num_samples_each_id = 1000

    num_blendshapes = 50
    output_dir_origin = 'blendshape_gt'
    if not os.path.exists(output_dir_origin):
        os.mkdir(output_dir_origin)

    for id in range(num_ids):
        output_filename = 'id_{}_{}.csv'.format(id, num_samples_each_id)
        print('writing {}'.format(output_filename))
        with open(os.path.join(output_dir_origin, output_filename), 'w') as f:
            writer = csv.writer(f)
            for i in tqdm(range(num_samples_each_id)):
                # random: [0., 0.5] - 0, 1, 2, 3, 4
                if id < 5:
                    value_weight = np.random.uniform(low=0., high=0.5, size=num_blendshapes).tolist()
                    value_weight = constraint(value_weight)

                    # origin bs
                    sample_name = '{}_{:0>6d}.jpg'.format(id, i)
                    value_weight_str = ["%.6f" % x for x in value_weight]
                    cur_row = [sample_name] + value_weight_str
                    writer.writerow(cur_row)
                # random: [0., 0.7] - 5, 6
                elif 5 <= id < 7:
                    value_weight = np.random.uniform(low=0., high=0.7, size=num_blendshapes).tolist()
                    value_weight = constraint(value_weight)

                    # origin bs
                    sample_name = '{}_{:0>6d}.jpg'.format(id, i)
                    value_weight_str = ["%.6f" % x for x in value_weight]
                    cur_row = [sample_name] + value_weight_str
                    writer.writerow(cur_row)

                # candidate - only one value - 7
                elif id == 7:
                    candidate_val = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

                    value_weight = [0. for _ in range(num_blendshapes)]
                    value_weight[i % num_blendshapes] = random.choice(candidate_val)

                    sample_name = '{}_{:0>6d}.jpg'.format(id, i)
                    value_weight_str = ["%.6f" % x for x in value_weight]
                    cur_row = [sample_name] + value_weight_str
                    writer.writerow(cur_row)
                # candidate - two values - 8
                elif id == 8:
                    candidate_val = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

                    value_weight = [0. for _ in range(num_blendshapes)]
                    index1 = i % num_blendshapes
                    index2 = random.randint(0, num_blendshapes - 1)
                    while index2 == index1:
                        index2 = random.randint(0, num_blendshapes - 1)
                    value_weight[index1] = random.choice(candidate_val)
                    value_weight[index2] = random.choice(candidate_val)

                    sample_name = '{}_{:0>6d}.jpg'.format(id, i)
                    value_weight_str = ["%.6f" % x for x in value_weight]
                    cur_row = [sample_name] + value_weight_str
                    writer.writerow(cur_row)
                # candidate - three values - 9
                else:
                    candidate_val = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

                    value_weight = [0. for _ in range(num_blendshapes)]
                    index1 = i % num_blendshapes
                    index2 = random.randint(0, num_blendshapes - 1)
                    while index2 == index1:
                        index2 = random.randint(0, num_blendshapes - 1)
                    index3 = random.randint(0, num_blendshapes - 1)
                    while index3 == index1 or index3 == index2:
                        index3 = random.randint(0, num_blendshapes - 1)

                    value_weight[index1] = random.choice(candidate_val)
                    value_weight[index2] = random.choice(candidate_val)
                    value_weight[(i + random.randint(1, num_blendshapes - 1)) % num_blendshapes] = random.choice(
                        candidate_val)

                    sample_name = '{}_{:0>6d}.jpg'.format(id, i)
                    value_weight_str = ["%.6f" % x for x in value_weight]
                    cur_row = [sample_name] + value_weight_str
                    writer.writerow(cur_row)


def csv_to_json(json_file="./blendshape_gt.json"):
    result = {}
    cnt = 0
    bs_files = "./blendshape_gt"
    bs_list = os.listdir(bs_files)
    for file in bs_list:
        file_path = os.path.join(bs_files, file)
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    key = "{:05d}.png".format(cnt)
                    values = row[1:]
                    result[key] = values
                    cnt += 1
    with open(json_file, 'w') as file:
        json.dump(result, file, indent=4)


if __name__ == '__main__':
    # generate()
    csv_to_json()
