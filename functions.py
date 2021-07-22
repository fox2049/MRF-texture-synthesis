# -*- coding: utf-8 -*-
# @Author  : fox2049
# @Time    : 2021/7/22 17:08
# @Function: functions for synthesize_textures.py

import numpy as np


def getNeighbourhood(padded_output_image, padded_filled_map, pix_row, pix_col,
                     window_size):
    half_window = window_size // 2
    pix_row = pix_row + half_window
    pix_col = pix_col + half_window
    neighbourhood = padded_output_image[
                    pix_row - half_window:pix_row + half_window+1, pix_col - half_window: pix_col + half_window+1, :]
    mask = padded_filled_map[
           pix_row - half_window:pix_row + half_window+1, pix_col - half_window: pix_col + half_window+1]
    return [neighbourhood, mask]


def ind2sub(array_shape, ind):
    rows = ind % array_shape[0] + 1
    cols = ind // array_shape[0] + 1
    return [rows, cols]


def im2double(im):
    info = np.iinfo(im.dtype)  # Get the data type of the input image
    return im.astype(np.float) / info.max  # Divide all values by the largest possible value in the datatype


def seed_image(input_image, output_w, output_h):
    width, height, channel = input_image.shape

    rand_w = np.random.randint(0, width - seed_size)
    rand_h = np.random.randint(0, height - seed_size)

    seed_patch = input_image[rand_w:rand_w + seed_size, rand_h:rand_h + seed_size, :]

    output_image = np.zeros([output_w, output_h, channel])

    center_w = output_w // 2
    center_h = output_h // 2
    half_seed_size = seed_size // 2
    output_image[center_w - half_seed_size:center_w + half_seed_size + 1,
    center_h - half_seed_size:center_h + half_seed_size + 1, :] = seed_patch

    filled_map = np.zeros([output_w, output_h])
    filled_map[center_w - half_seed_size:center_w + half_seed_size + 1,
    center_h - half_seed_size:center_h + half_seed_size + 1] = 1

    return [output_image, filled_map]


def im2col(mtx, block_size):
    mtx_shape = mtx.shape
    sx = mtx_shape[0] - block_size[0] + 1
    sy = mtx_shape[1] - block_size[1] + 1
    # 如果设A为m×n的，对于[p q]的块划分，最后矩阵的行数为p×q，列数为(m−p+1)×(n−q+1)。
    result = np.empty((block_size[0] * block_size[1], sx * sy))
    # 沿着行移动，所以先保持列（i）不动，沿着行（j）走
    for i in range(sy):
        for j in range(sx):
            result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].ravel(order='F')

    return result


def copyMakeBorder(f):
    rows, cols = f.shape
    newF = np.zeros((rows + 2, cols + 2))
    newF[0, 1:-1] = f[0, :]
    newF[1:-1, 1:-1] = f[0:, :]
    newF[rows, 1:-1] = f[rows - 1:]
    newF[:, 0] = newF[:, 1]
    newF[:, -1] = newF[:, -2]
    return newF


def imdilate(f, B=np.ones(9).reshape(3, 3)):
    rowsB, colsB = B.shape
    rb = int(rowsB / 2)
    cb = int(colsB / 2)
    f = copyMakeBorder(f)
    newF = np.ndarray(f.shape)
    rows, cols = f.shape
    for i in range(rb, rows - rb):
        for j in range(cb, cols - cb):
            newF[i, j] = 1 if np.sum(f[i - rb:i + rb + 1, j - cb:j + cb + 1] * B) >= 1 else 0
    return newF[1:-1, 1:-1]


def getUnfilledPixel(filled_map):
    dilated_map = imdilate(filled_map)
    diff_image = dilated_map - filled_map
    unfilled_pixels = np.argwhere(diff_image != 0)
    return unfilled_pixels