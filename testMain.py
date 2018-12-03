

import numpy as np
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import timeit

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
CHANNEL_NUM = 3


def cal_dir_stat(root):
    lines = [line.rstrip('\n') for line in open(root)]
    paths = [line.split()[0] for line in lines]
    cls_dirs = [d for d in listdir(root) if isdir(join(root, d))]
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    for idx, d in enumerate(cls_dirs):
        print("#{} class".format(idx))
        im_pths = glob(join(root, d, "*.jpg"))

        for path in im_pths:
            im = cv2.imread(path) # image in M*N*3 shape, channel in BGR order
            im = im/255.0
            pixel_num += (im.size/3)
            channel_sum += np.sum(im, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]
    return rgb_mean, rgb_std




def main():
    # lines = [line.rstrip('\n') for line in open('./origaAll.txt')]
    # paths = [line.split()[0] for line in lines]
    # labels = []
    # [labels.append(line.split()[1]) for line in lines]
    # # for line in lines:
    # #   labels.append(line.split()[1])
    # The script assumes that under train_root, there are separate directories for each class
    # of training images.
    train_root = "./origaAll.txt"
    start = timeit.default_timer()
    mean, std = cal_dir_stat(train_root)
    end = timeit.default_timer()
    print("elapsed time: {}".format(end - start))
    print("mean:{}\nstd:{}".format(mean, std))
    # print(lines)


if __name__ == '__main__':
    main()