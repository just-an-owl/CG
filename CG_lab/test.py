from PIL import Image as im
import numpy as np


def test():
    h = 128
    w = 128

    matrix = np.zeros((h, w), dtype=np.uint8)  # uint8 it's unsigned 8-bit integer (0, 255)
    data = im.fromarray(matrix)
    data.save(f'./CG_lab1/images/{h}x{w}_1ch_(0)_image.png')

    matrix -= 1  # 255 <- 0
    data = im.fromarray(matrix)
    data.save(f'./CG_lab1/images/{h}x{w}_1ch_(255)_image.png')

    rgb_matrix = np.zeros((h, w, 3), dtype=np.uint8)  # [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
    rgb_matrix[:, :, 0] = 255  # [[[255, 0, 0], [255, 0, 0]], [[255, 0, 0], [255, 0, 0]], [[255, 0, 0], [255, 0, 0]]]
    data = im.fromarray(rgb_matrix)
    data.save(f'./CG_lab1/images/{h}x{w}_3ch_(255,0,0)_image.png')  # rgb -> [255, 0, 0] -> red image

    rgb_random_matrix = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    print(f"{rgb_random_matrix.shape=}")
    data = im.fromarray(rgb_random_matrix)
    data.save(f'./CG_lab1/images/{h}x{w}_3ch_random_image.png')
