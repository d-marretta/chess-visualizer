import json
import main
from main import get_coords_chessboards, warp_chessboard, get_bboxes, get_fen_board
import ultralytics
import os



def compare_boards(true_board, predicted_board):
    match_count = 0
    total_positions = 64

    for i in range(8):
        for j in range(8):
            if true_board[i][j] == predicted_board[i][j]:
                match_count += 1

    errors = total_positions - match_count

    return errors

def evaluate_chessboard_function(seg_yolo, detect_yolo, dataset_path, allowed_errors=0):
    correct_count = 0
    total_images = 0
    
    for filename in os.listdir(f'{dataset_path}/images'):
        name = filename.split('.')[0]

        with open(f'{dataset_path}/labels/{name}.txt', 'r') as f:
            true_fen = f.read().strip()
        
        true_board = fen_to_matrix(true_fen)

        for orientation in ['north', 'east', 'west', 'south']:
            try:
                predicted_board = get_chessboard_positions(seg_yolo, detect_yolo, f'{dataset_path}/images/{filename}', orientation)
            except:
                continue

            errors = compare_boards(true_board, predicted_board)

            if errors <= allowed_errors:
                correct_count += 1
                break            

        total_images += 1

    accuracy = correct_count / total_images if total_images > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total_images})")
    return accuracy

def fen_to_matrix(fen):
    board = []
    rows = fen.split(' ')[0].split('/') 

    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend(['.'] * int(char)) 
            else:
                board_row.append(char) 
        board.append(board_row)

    return board

def get_chessboard_positions(seg_yolo, detect_yolo, image_path, white_orientation):
    vertices = get_coords_chessboards(seg_yolo, image_path)
    M, square_size = warp_chessboard(vertices)
    positions = get_bboxes(detect_yolo, image_path, M, square_size)
    board = get_fen_board(positions, white_orientation)
    fen = board.board_fen()
    matrix = fen_to_matrix(fen)

    return matrix
    
if __name__ == '__main__':
    seg_yolo = ultralytics.YOLO('./models/yolo11n-seg-best-new.pt')
    detect_yolo = ultralytics.YOLO('./models/yolo11n-best-.pt')
    evaluate_chessboard_function(seg_yolo, detect_yolo, './datasets/test_dataset', allowed_errors=0)
