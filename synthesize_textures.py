# -*- coding: utf-8 -*-
# @Author  : fox2049
# @Time    : 2021/7/22 17:10
# @Function: synthesize_textures

from cv2 import cv2
import matplotlib.pyplot as plt
from functions import *


def synthesize_texture(input_image, output_w, output_h, window_size):
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2

    width, height, channel = input_image.shape

    [output_image, filled_map] = seed_image(input_image, output_w, output_h)
    padded_output_image = np.pad(output_image, ((half_window, half_window), (half_window, half_window), (0, 0)))
    padded_filled_map = np.pad(filled_map, ((half_window, half_window), (half_window, half_window)))

    error_threshold = 0.1
    max_error_threshold = 0.3

    num_horiz_candidates = width - window_size + 1
    num_vert_candidates = height - window_size + 1

    candidates = np.zeros([window_size ** 2, num_horiz_candidates * num_vert_candidates, channel])

    for c in range(channel):
        test_out = im2col(input_image[:, :, c], [window_size, window_size])
        candidates[:, :, c] = test_out

    permuted_candidates = np.transpose(candidates, [0, 2, 1])
    stacked_candidate_channels = permuted_candidates.reshape([-1, np.size(candidates, 1)], order="F")

    sigma = 6.4
    gaussian = np.multiply(cv2.getGaussianKernel(window_size, window_size / sigma),
                           (cv2.getGaussianKernel(window_size, window_size / sigma)).T)

    gaussian_vec = gaussian.reshape([-1, 1])
    gaussian_vec = np.tile(gaussian_vec, [np.size(candidates, 2), 1])

    while 0 in filled_map:
        found_match = 0

        unfilled_pixels = getUnfilledPixel(filled_map)

        for [pix_col, pix_row] in unfilled_pixels:

            [neighbourhood, mask] = getNeighbourhood(padded_output_image, padded_filled_map, pix_row, pix_col,
                                                     window_size)
            neighbourhood_vec = neighbourhood.reshape([-1, 1], order="F")
            neighbourhood_rep = np.tile(neighbourhood_vec, [1, np.size(candidates, 1)])

            mask_vec = mask.reshape([-1, 1], order="F")
            mask_vec = np.tile(mask_vec, [np.size(candidates, 2), 1])

            weight = sum(mask_vec * gaussian_vec)

            gaussian_mask = ((gaussian_vec * mask_vec) / weight).T
            distances = gaussian_mask.dot(np.power((stacked_candidate_channels - neighbourhood_rep), 2))
            min_value = min(distances[0])

            min_threshold = min_value * (1 + error_threshold)

            min_positions = np.argwhere(distances[0] <= min_threshold)
            random_col = np.random.randint(len(min_positions))
            selected_patch = min_positions[random_col]
            selected_error = distances[0, selected_patch]

            if selected_error < max_error_threshold:
                [matched_row, matched_col] = ind2sub([(width - window_size + 1), (height - window_size + 1)],
                                                     selected_patch)

                matched_row = matched_row + half_window
                matched_col = matched_col + half_window

                output_image[pix_row, pix_col, :] = input_image[matched_row, matched_col, :]

                filled_map[pix_row, pix_col] = 1

                found_match = 1
        plt.imshow(output_image)

        padded_output_image[half_window + 1:half_window + output_w + 1, half_window + 1:half_window + output_h + 1,
        :] = output_image
        padded_filled_map[
        half_window + 1: half_window + output_w + 1, half_window + 1: half_window + output_h + 1] = filled_map

        if not found_match:
            max_error_threshold *= 1.1
    return output_image


if __name__ == '__main__':
    input_image = cv2.imread(r"\text3.png")
    input_image = im2double(input_image)
    output_image = synthesize_texture(input_image, 128, 128, 21)
    plt.savefig("filename.png")
    plt.show()
