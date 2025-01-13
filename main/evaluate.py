import json
import main
from main import get_coords_chessboards, warp_chessboard, get_bboxes, get_fen_board
import ultralytics

def true_positions_to_matrix(annotations, image_id):
    piece_id_to_symbol = {  0:"P",
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
                            11:"k" }
    
    board = [["." for _ in range(8)] for _ in range(8)]
    for piece in annotations['pieces']:
        if piece['image_id'] == image_id:
            pos = piece['chessboard_position']
            row = 8 - int(pos[1])
            col = ord(pos[0]) - ord('a')
            board[row][col] = piece_id_to_symbol[piece['category_id']]
    return board

def compare_boards(true_board, predicted_board):
    match_count = 0
    total_positions = 64

    for i in range(8):
        for j in range(8):
            if true_board[i][j] == predicted_board[i][j]:
                match_count += 1

    errors = total_positions - match_count

    return errors

def evaluate_chessboard_function(seg_yolo, detect_yolo, json_file_path, allowed_errors=0):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    correct_count = 0
    total_images = 0
    test_ids = data['splits']['chessred2k']['test']['image_ids']
    for image in data['images']:
        right = False
        image_id = image['id']
        if not (image_id in test_ids):
            continue
        image_path = f'./backup/{image['path']}'
        true_board = true_positions_to_matrix(data['annotations'], image_id)

        for orientation in ['north', 'east', 'west', 'south']:
            try:
                predicted_board = get_chessboard_positions(seg_yolo, detect_yolo, image_path, orientation)
            except:
                continue

            errors = compare_boards(true_board, predicted_board)

            if errors <= allowed_errors:
                correct_count += 1
                right = True
                break            

        total_images += 1
        # if not right:
        #     print(image_path)

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
    seg_yolo = ultralytics.YOLO('./models/yolo11n-seg-best.pt')
    detect_yolo = ultralytics.YOLO('./models/yolo11n-best.pt')
    evaluate_chessboard_function(seg_yolo, detect_yolo, './backup/annotations.json', allowed_errors=0)
