# # import numpy as np


# # def select_move(cur_state, remain_time):
# #     valid_moves = cur_state.get_valid_moves
# #     if len(valid_moves) != 0:
# #         return np.random.choice(valid_moves)
# #     return None

# import numpy as np
# from state import State, State_2

import numpy as np
from state import State, State_2

# def select_move(cur_state, remain_time):
#     st=State_2(cur_state)
#     valid_moves = st.get_valid_moves  # Đây là thuộc tính, không cần dấu ngoặc
#     if len(valid_moves) != 0:
#         if not valid_moves:
#             return None

#         best_move = valid_moves[0]
#         best_score = float('-inf')  # Negative infinity
#         alpha = float('-inf')
#         beta = float('inf')

#         for move in valid_moves:
#             new_state = State_2(st)
#             new_state.act_move(move)
#             score = minimax_alpha_beta(new_state, depth=3, alpha=alpha, beta=beta, maximizing_player=False)

#             if score > best_score:
#                 best_score = score
#                 best_move = move

#             alpha = max(alpha, best_score)

#         # cur_state.act_move(best_move)
#         return best_move


# def minimax_alpha_beta(state, depth, alpha, beta, maximizing_player):
#     if depth == 0 or state.game_over:
#         return evaluate_state(state)

#     valid_moves = state.get_valid_moves

#     if not valid_moves:
#         # Không có bước đi hợp lệ, trả về giá trị đánh giá cho trạng thái hiện tại
#         return evaluate_state(state)

#     if maximizing_player:
#         max_eval = float('-inf')
#         for move in valid_moves:
#             new_state = State_2(state)
#             new_state.act_move(move)
#             eval = minimax_alpha_beta(new_state, depth - 1, alpha, beta, False)
#             max_eval = max(max_eval, eval)
#             alpha = max(alpha, max_eval)
#             if beta <= alpha:
#                 break
#         return max_eval
#     else:
#         min_eval = float('inf')
#         for move in valid_moves:
#             new_state = State_2(state)
#             new_state.act_move(move)
#             eval = minimax_alpha_beta(new_state, depth - 1, alpha, beta, True)
#             min_eval = min(min_eval, eval)
#             beta = min(beta, min_eval)
#             if beta <= alpha:
#                 break
#         return min_eval
# def evaluate_state(state):
#     result = state.game_result(state.global_cells.reshape(3, 3))

#     if result == state.X:
#         return 1
#     elif result == state.O:
#         return -1
#     else:
#         total_score = 0

#         # Evaluate each small local board
#         for i in range(9):
#             local_board = state.blocks[i]
#             local_result = state.game_result(local_board)

#             if local_result == state.X:
#                 total_score += 1
#             elif local_result == state.O:
#                 total_score -= 1

#         return total_score

def select_move(cur_state, remain_time):
    count=len((np.where(cur_state.global_cells == 0))[0])
    valid_moves = cur_state.get_valid_moves

    if not valid_moves:
        return False  

    best_move = None
    best_score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')

    for move in valid_moves:
        # Tạo một bản sao của trạng thái hiện tại để thực hiện nước đi
        new_state = State(cur_state)
        new_state.act_move(move)

        # Chạy thuật toán minimax để đánh giá nước đi
        score = mini_max(new_state, min(1, count), alpha, beta, maximizing_player=False)


        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move

def evaluate_game(state):
    evale = 0
    main_bd = np.zeros(9)
    evaluator_mul = [1.4, 1, 1.4, 1, 1.75, 1, 1.4, 1, 1.4]
    currentBoard = 9
    if state.previous_move != None:
        currentBoard = state.previous_move.x * 3 + state.previous_move.y
    # Loop through each element in the position array
    for eh in range(9):
        evale += real_evaluate_square(state, eh) * 1.5 * evaluator_mul[eh]

        # If the current element is the same as the current board, add its evaluation to the total
        if eh == currentBoard:
            evale += real_evaluate_square(state, eh) * evaluator_mul[eh]

        # Check for winning condition in the current square and subtract its evaluation from the total
        tmp_ev = state.game_result(state.blocks[eh])
        if tmp_ev is not None:
            evale -= tmp_ev * evaluator_mul[eh]

        # Store the evaluation of the current square in the main_bd list
        main_bd[eh]= tmp_ev

    # Subtract the overall winning condition evaluation from the total
    if state.game_result(main_bd.reshape(3,3)) != None:
        evale -= state.game_result(main_bd.reshape(3,3)) * 5000

    # Add the evaluation of the main_bd to the total
    evale += real_evaluate_square(state, currentBoard) * 150

    return evale

# Placeholder functions - you need to implement these according to your requirements
def real_evaluate_square(state, index):
    # Implement the real evaluation logic for a single square
    pos_2d = state.blocks[index]
    evaluation = 0
    points = np.array([0.2, 0.17, 0.2, 0.17, 0.22, 0.17, 0.2, 0.17, 0.2])

    for i in range(3):
        for j in range(3):
            evaluation -= pos_2d[i, j] * points[i * 3 + j]

    a = 2
    if np.any(np.sum(pos_2d, axis=0) == a):
        evaluation -= 6

    if np.any(np.sum(pos_2d, axis=1) == a):
        evaluation -= 6

    if np.trace(pos_2d) == a or np.trace(np.flipud(pos_2d)) == a:
        evaluation -= 7

    a = -1
    if (np.sum(pos_2d[:2, :], axis=0) == 2 * a).any() and (pos_2d[2, 2] == -a).all() or \
    (np.sum(pos_2d[1:, :], axis=0) == 2 * a).any() and (pos_2d[0, 0] == -a).all() or \
    (np.sum(pos_2d[:2, :], axis=0) == 2 * a).any() and (pos_2d[0, 1] == -a).all() or \
    (np.sum(pos_2d[:, :2], axis=1) == 2 * a).any() and (pos_2d[2, 2] == -a).all() or \
    (np.sum(pos_2d[:, 1:], axis=1) == 2 * a).any() and (pos_2d[0, 0] == -a).all() or \
    (np.sum(pos_2d[:, :2], axis=1) == 2 * a).any() and (pos_2d[1, 2] == -a).all() or \
    (np.sum(pos_2d[:3, 0]) == 2 * a).any() and (pos_2d[2, 0] == -a).all() or \
    (np.sum(pos_2d[:3, 1]) == 2 * a).any() and (pos_2d[2, 1] == -a).all() or \
    (np.sum(pos_2d[:3, 2]) == 2 * a).any() and (pos_2d[2, 2] == -a).all() or \
    (np.sum(pos_2d[0, :3]) == 2 * a).any() and (pos_2d[0, 2] == -a).all() or \
    (np.sum(pos_2d[1, :3]) == 2 * a).any() and (pos_2d[1, 2] == -a).all() or \
    (np.sum(pos_2d[2, :3]) == 2 * a).any() and (pos_2d[2, 2] == -a).all() or \
    (np.sum(np.diagonal(pos_2d)) == 2 * a).any() and (pos_2d[2, 2] == -a).all() or \
    (np.sum(np.diagonal(np.flipud(pos_2d))) == 2 * a).any() and (pos_2d[2, 0] == -a).all() or \
    (np.sum(np.diagonal(pos_2d)) == 2 * a).any() and (pos_2d[0, 2] == -a).all() or \
    (np.sum(np.diagonal(np.flipud(pos_2d))) == 2 * a).any() and (pos_2d[0, 0] == -a).all():
        evaluation -= 9

    a = -2
    if np.any(np.sum(pos_2d, axis=0) == a):
        evaluation += 6

    if np.any(np.sum(pos_2d, axis=1) == a):
        evaluation += 6

    if np.trace(pos_2d) == a or np.trace(np.flipud(pos_2d)) == a:
        evaluation += 7

    a = 1
    if (np.sum(pos_2d[:2, :]) == 2 * a).any() and (pos_2d[2, 2] == -a).all() or \
    (np.sum(pos_2d[1:, :]) == 2 * a).any() and (pos_2d[0, 0] == -a).all() or \
    (np.sum(pos_2d[:2, :]) == 2 * a).any() and (pos_2d[0, 1] == -a).all() or \
    (np.sum(pos_2d[:, :2]) == 2 * a).any() and (pos_2d[2, 2] == -a).all() or \
    (np.sum(pos_2d[:, 1:]) == 2 * a).any() and (pos_2d[0, 0] == -a).all() or \
    (np.sum(pos_2d[:, :2]) == 2 * a).any() and (pos_2d[1, 2] == -a).all() or \
    (np.sum(pos_2d[:3, 0]) == 2 * a).any() and (pos_2d[2, 0] == -a).all() or \
    (np.sum(pos_2d[:3, 1]) == 2 * a).any() and (pos_2d[2, 1] == -a).all() or \
    (np.sum(pos_2d[:3, 2]) == 2 * a).any() and (pos_2d[2, 2] == -a).all() or \
    (np.sum(pos_2d[0, :3]) == 2 * a).any() and (pos_2d[0, 2] == -a).all() or \
    (np.sum(pos_2d[1, :3]) == 2 * a).any() and (pos_2d[1, 2] == -a).all() or \
    (np.sum(pos_2d[2, :3]) == 2 * a).any() and (pos_2d[2, 2] == -a).all() or \
    (np.sum(np.diagonal(pos_2d)) == 2 * a).any() and (pos_2d[2, 2] == -a).all() or \
    (np.sum(np.diagonal(np.flipud(pos_2d))) == 2 * a).any() and (pos_2d[2, 0] == -a).all() or \
    (np.sum(np.diagonal(pos_2d)) == 2 * a).any() and (pos_2d[0, 2] == -a).all() or \
    (np.sum(np.diagonal(np.flipud(pos_2d))) == 2 * a).any() and (pos_2d[0, 0] == -a).all():
        evaluation += 9

    result = state.game_result(state.blocks[index])
    if result is not None:
        evaluation -= result * 12


    return evaluation


def mini_max(state, depth, alpha, beta, maximizing_player):

    calc_eval = evaluate_game(state)
    if depth <= 0 or abs(calc_eval) > 5000:
        return calc_eval

    valid_moves = state.get_valid_moves
    
    if maximizing_player == True:
        max_eval = float('-inf')
        for move in valid_moves:
            new_state=State(state)
            new_state.act_move(move) 
            evalut = mini_max(new_state, depth - 1, alpha,  beta, False)
            max_eval = max(max_eval, evalut)
            alpha = max(alpha, evalut)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in valid_moves:
            new_state=State(state)
            new_state.act_move(move)
            evalua = mini_max(new_state, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, evalua)
            beta = min(beta, evalua)
            if beta <= alpha:
                break
        return min_eval