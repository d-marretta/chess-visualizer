import chess.svg
import cv2
import numpy as np
import math
import chess
import cairosvg
import ultralytics

def get_coords_chessboards(model, image_path):
    results = model(image_path, verbose=False)
    result = results[0]
    contour = result.masks.xy.pop()

    contour = contour.astype(np.int32)

    contour = contour.reshape(-1, 1, 2)
    epsilon = 0.02*cv2.arcLength(contour, False)
    vertices = cv2.approxPolyDP(contour, epsilon, closed=True)  
    vertices = np.squeeze(vertices)
    vertices = order_vertices(vertices)
    
    return vertices
    

def order_vertices(vertices):
    sorted_vertices = sorted(vertices, key=lambda v: (-v[0],v[1]))
    right_vertices = sorted_vertices[:2]
    left_vertices = sorted_vertices[2:]

    top_right = min(right_vertices, key=lambda v: v[1])
    bottom_right = max(right_vertices, key=lambda v: v[1])

    top_left = min(left_vertices, key=lambda v: v[1])
    bottom_left = max(left_vertices, key=lambda v: v[1])

    return top_left, top_right, bottom_right, bottom_left

def warp_chessboard(vertices):
    top_left, top_right, bottom_right, bottom_left = vertices
    n_squares = 8

    side_chessboard = int(math.sqrt((top_right[0] - top_left[0]) ** 2 + (top_right[1] - top_right[1]) ** 2))

    input_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
    dest_pts = np.float32([[0,0], [side_chessboard, 0], [side_chessboard, side_chessboard], [0, side_chessboard] ])

    M = cv2.getPerspectiveTransform(input_pts, dest_pts)
    square_size = side_chessboard / n_squares

    return M, square_size


def get_bboxes(model, image_path, M, square_size):
    classes = { 0:"P",
                1:"R",
                2:"N",
                3:"B",
                4:"Q",
                5:"K",
                6:"p",
                7:"r",
                8:"n",
                9:"b",
                10:"q",
                11:"k"}
    
    results = model(image_path,conf=0.30, verbose=False)
    result = results[0]
    bboxs = []
    for box in result.boxes:
        b = box.xywh[0]
        b_class = classes[int(box.cls)]
        bboxs.append((b_class, b))

    original_points = np.array([[[bbox[1][0], bbox[1][1]+bbox[1][3]//2.4]] for bbox in bboxs], dtype='float32')

    transformed_points = cv2.perspectiveTransform(original_points, M)

    positions = []
    for i, point in enumerate(transformed_points):
        transformed_x, transformed_y = point[0][0], point[0][1]    
        
        square_col = int(transformed_x // square_size)
        square_row = int(transformed_y // square_size)+1
        piece_pos = (8-square_row, square_col, bboxs[i][0])
        positions.append(piece_pos)

    positions.sort(key=lambda x: (-x[0],x[1]))

    return positions

def get_fen_board(positions, white_orientation):
    board = [["" for _ in range(8)] for _ in range(8)]

    for row, col, piece_type in positions:
        board[row][col] = piece_type

    fen_rows = []
    for row in reversed(board):
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == "":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    fen_board = "/".join(fen_rows)
    board = chess.Board(fen_board)

    if white_orientation == 'west':
        board = board.transform(chess.flip_diagonal)
        board = board.transform(chess.flip_horizontal)
    elif white_orientation == 'north':
        board = board.transform(chess.flip_vertical)
        board = board.transform(chess.flip_horizontal)
    elif white_orientation == 'east':
        board = board.transform(chess.flip_anti_diagonal)
        board = board.transform(chess.flip_horizontal)

    return board

def main(seg_yolo, detect_yolo, image_path, white_orientation = 'south'):
    vertices = get_coords_chessboards(seg_yolo, image_path)
    M, square_size = warp_chessboard(vertices)
    positions = get_bboxes(detect_yolo, image_path, M, square_size)
    board = get_fen_board(positions, white_orientation)

    cairosvg.svg2png(chess.svg.board(board, size=1200),write_to='prova.png')


if __name__ == '__main__':
    seg_yolo = ultralytics.YOLO('./models/yolo11n-seg-best.pt')
    detect_yolo = ultralytics.YOLO('./models/yolo11n-best.pt')
    image_path = ''
    main(seg_yolo, detect_yolo, image_path, 'west')
    

