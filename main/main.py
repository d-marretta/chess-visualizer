import cv2
import os
import numpy as np
import math
im = cv2.imread('datasets/dataset/images/train/G019_IMG014.jpg')
height, width, _ = im.shape
#coords = [0.8166071428571429, 0.4472850529100529, 0.5936044973544974, 0.7356349206349206, 0.1873776455026455, 0.5298181216931217, 0.4569113756613757, 0.3193915343915344]
coords = [0.22916997354497354, 0.7538161375661375, 0.16593915343915344, 0.31415674603174604, 0.6678042328042328, 0.24065806878306878, 0.8885284391534392, 0.6222685185185185]

top_left = coords[0] * width, coords[1] * height
top_right = coords[2] * width, coords[3] * height
bottom_right = coords[4] * width, coords[5] * height
bottom_left = coords[6] * width, coords[7] * height

side_chessboard = int(math.sqrt((top_right[0] - top_left[0]) ** 2 + (top_right[1] - top_right[1]) ** 2))

input_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
dest_pts = np.float32([[0,0], [side_chessboard, 0], [side_chessboard, side_chessboard], [0, side_chessboard] ])

M = cv2.getPerspectiveTransform(input_pts, dest_pts)

warped = cv2.warpPerspective(im, M, (side_chessboard, side_chessboard))

n_squares = 8
square_size = side_chessboard / n_squares

grid_lines = warped.copy()

for i in range(n_squares + 1):
    y = int(i * square_size)
    cv2.line(grid_lines, (0, y), (side_chessboard, y), (0, 0, 255), 2)

for i in range(n_squares + 1):
    x = int(i * square_size)
    cv2.line(grid_lines, (x, 0), (x, side_chessboard), (0, 0, 255), 2)  

cv2.imwrite('prova.jpg', grid_lines)


