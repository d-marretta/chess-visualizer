import chess
import chess.svg

def render_chessboard(fen_board, white_orientation):
    board = chess.Board(fen_board)

    if white_orientation == 'left':
        board = board.transform(chess.flip_diagonal)
        board = board.transform(chess.flip_horizontal)
    elif white_orientation == 'top':
        board = board.transform(chess.flip_vertical)
        board = board.transform(chess.flip_horizontal)
    elif white_orientation == 'right':
        board = board.transform(chess.flip_anti_diagonal)
        board = board.transform(chess.flip_horizontal)

    return chess.svg.board(board, size=1200)
