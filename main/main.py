import cv2
import os
import numpy as np
import math
im = cv2.imread('datasets/detection_dataset/images/test/G000_IMG000.jpg')
height, width, _ = im.shape
coords = [0.15908203125, 0.35113932291666666, 0.5768977864583333, 0.20787434895833334, 0.84970703125, 0.5081054687500001, 0.34612630208333334, 0.7500325520833333]

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
    cv2.line(grid_lines, (0, y), (side_chessboard, y), (0, 0, 255), 1)

for i in range(n_squares + 1):
    x = int(i * square_size)
    cv2.line(grid_lines, (x, 0), (x, side_chessboard), (0, 0, 255), 1)  


bbox = [0.41768229166666665, 0.24734700520833333, 0.044514973958333336, 0.09892578125]

x_center, y_center, box_width, box_height = bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height
original_point = np.array([[x_center, y_center+box_height//2.3]], dtype='float32')  # Shape (1, 1, 2)
original_point = np.array([original_point])  # Adding batch dimension for cv2.perspectiveTransform

transformed_point = cv2.perspectiveTransform(original_point, M)

transformed_x, transformed_y = transformed_point[0][0][0], transformed_point[0][0][1]

cv2.circle(grid_lines, (int(transformed_x),int(transformed_y)), 3, (0,0,255), -1)
cv2.imwrite('prova.jpg', grid_lines)

