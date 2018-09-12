# _*_coding:utf-8 _*_
# Author  : Tao
"""
用于加载tvsum anno数据
"""

import os
import cv2
import csv
import numpy as np


def load_tvsum(data_root_path, scale=False, ):
    """
    :param data_root_path: tvsum data path
    :param scale: scale to 0.0 ~ 1.0 or not
    :return: dic
    """
    anno_file = os.path.join(data_root_path, 'data/ydata-tvsum50-anno.tsv')
    video_path = os.path.join(data_root_path, 'video')

    video_name, fps, nframes, user_score, avg_score = [], [], [], [], []

    with open(anno_file) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for us in rd:  # user summary
            if us[0] not in video_name:
                video_name.append(us[0])
                vidx = video_name.index(us[0])
                capture = cv2.VideoCapture(
                    os.path.join(video_path, us[0] + '.mp4'))

                # get fps
                fps.append(int(capture.get(cv2.CAP_PROP_FPS)))
                # get frames
                nframes.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                user_score.append([])
            user_score[vidx].append(np.asarray(us[2].split(',')).astype(float))

        for vidx in range(len(video_name)):  # video key
            avg_score.append(np.asarray(user_score[vidx]).mean(axis=0))
            user_score[vidx] = np.asarray(user_score[vidx])

    # scale to 0.0 ~ 1.0
    if scale:
        for i in range(len(avg_score)):
            max_score = max(avg_score[i])
            min_score = min(avg_score[i])
            avg_s = (avg_score[i] - min_score) / (max_score - min_score)
            avg_score[i] = avg_s

    dic = {}
    for i, video in enumerate(video_name):
        dic[video] = {'fps': fps[i],
                      'frames': nframes[i],
                      'user_score': user_score[i],
                      'avg_score': avg_score[i]
                      }
    return dic


def knapsack_dp(values, weights, n_items, capacity, return_all=False):
    check_inputs(values, weights, n_items, capacity)

    table = np.zeros((n_items+1, capacity+1), dtype=np.float32)
    keep = np.zeros((n_items+1, capacity+1), dtype=np.float32)

    for i in range(1, n_items+1):
        for w in range(0, capacity+1):
            wi = weights[i-1]    # weight of current item
            vi = values[i-1]     # value of current item
            if (wi <= w) and (vi + table[i-1, w-wi] > table[i-1, w]):
                table[i, w] = vi + table[i-1, w-wi]
                keep[i, w] = 1
            else:
                table[i, w] = table[i-1, w]

    picks = []
    K = capacity

    for i in range(n_items, 0, -1):
        if keep[i, K] == 1:
            picks.append(i)
            K -= weights[i-1]

    picks.sort()
    picks = [x-1 for x in picks]     # change to 0-index

    if return_all:
        max_val = table[n_items, capacity]
        return picks, max_val
    return picks


def check_inputs(values, weights, n_items, capacity):
    # check variable type
    assert(isinstance(values, list))
    assert(isinstance(weights, list))
    assert(isinstance(n_items, int))
    assert(isinstance(capacity, int))
    # check value type
    assert(all(isinstance(val, int) or isinstance(val, float) for val in
               values))
    assert(all(isinstance(val, int) for val in weights))
    # check validity of value
    assert(all(val >= 0 for val in weights))
    assert(n_items > 0)
    assert(capacity > 0)


def get_summary(score, sum_rate):
    """
    :param score: score list
    :param sum_rate: summary rate
    :return: summary mask, one hot
    """
    clip_scores = [x * 1000 for x in score]  # up scale
    clip_scores = [int(round(x)) for x in clip_scores]

    n = len(clip_scores)  # 总帧数
    W = int(n * sum_rate)  # summary帧总数
    val = clip_scores  #
    wt = [1 for x in range(n)]  # 全1

    sum_ = knapsack_dp(val, wt, n, W)

    summary = np.zeros((1), dtype=np.float32)  # this element should be deleted
    for seg_idx in range(n):
        nf = wt[seg_idx]
        if seg_idx in sum_:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))
    summary = list(summary)

    del summary[0]

    return summary