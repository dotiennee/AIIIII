import numpy as np
from state import UltimateTTT_Move


class GameState:
    x = 1
    o = -1

    def __init__(self, state=None):  # init with 0 arg or 1 arg (state)
        if not state:
            self.global_cells = np.zeros(9)
            self.blocks = np.array([np.zeros((3, 3)) for x in range(9)])
            self.player_to_move: int = 1
            self.previous_move: UltimateTTT_Move is None
            self.free_move = False
        else:
            self.global_cells = np.copy(state.global_cells)
            self.blocks = np.copy(state.blocks)
            self.player_to_move: int = state.player_to_move
            self.previous_move: UltimateTTT_Move = state.previous_move
            self.free_move = state.free_move

    def __repr__(self):
        return '''player: {0} \n\nmove: {1} \n\nafter move:\n\n+ global cells: 
                \n\n{2}\n+ blocks:\n\n{3}
                \n#############################################\n'''.format(
            self.player_to_move * -1,
            self.previous_move,
            self.global_cells.reshape(3, 3),
            self.blocks
        )

    # game result on single board (local or global)
    def game_result(self, board):
        row_sum = np.sum(board, 1)
        col_sum = np.sum(board, 0)
        diag_sum_top_left = board.trace()
        diag_sum_top_right = board[::-1].trace()

        player_one_wins = any(row_sum == 3) + any(col_sum == 3)
        player_one_wins += (diag_sum_top_left == 3) + (diag_sum_top_right == 3)

        if player_one_wins:
            return self.x

        player_two_wins = any(row_sum == -3) + any(col_sum == -3)
        player_two_wins += (diag_sum_top_left == -3) + (diag_sum_top_right == -3)

        if player_two_wins:
            return self.o

        if np.all(board != 0):
            return 0

        # if not over
        return None

    @property
    def game_over(self):
        return self.game_result(self.global_cells.reshape(3, 3)) is not None

    @property
    def get_valid_moves(self):
        if self.previous_move is not None:
            index_local_board = self.previous_move.x * 3 + self.previous_move.y
        else:
            temp_blocks = np.zeros((3, 3))
            indices = np.where(temp_blocks == 0)
            ret = []
            for i in range(9):
                ret += [UltimateTTT_Move(i, index[0], index[1], self.player_to_move)
                        for index in list(zip(indices[0], indices[1]))
                        ]
            return ret

        local_board = self.blocks[index_local_board]
        indices = np.where(local_board == 0)

        if len(indices[0]) != 0:
            self.free_move = False
            return [UltimateTTT_Move(index_local_board, index[0],
                                     index[1], self.player_to_move)
                    for index in list(zip(indices[0], indices[1]))
                    ]
        # chosen board is full
        self.free_move = True
        ret = []
        for i in range(9):
            if not np.all(self.blocks[i] != 0):
                indices = np.where(self.blocks[i] == 0)
                ret += [UltimateTTT_Move(i, index[0], index[1], self.player_to_move)
                        for index in list(zip(indices[0], indices[1]))
                        ]
        return ret

    def act_move(self, move: UltimateTTT_Move):
        local_board = self.blocks[move.index_local_board]
        local_board[move.x, move.y] = move.value

        self.player_to_move *= -1
        self.previous_move = move

        if self.global_cells[move.index_local_board] == 0:  # not 'X' or 'O'
            if self.game_result(local_board):
                self.global_cells[move.index_local_board] = move.value

        # print(self)


class GameState2(GameState):
    def __init__(self, state=None):
        super().__init__(state)

    @property
    def get_valid_moves(self):
        is_occupied = False
        if self.previous_move is not None:
            index_local_board = self.previous_move.x * 3 + self.previous_move.y
            if self.global_cells[index_local_board] != 0:
                is_occupied = True
        else:
            temp_blocks = np.zeros((3, 3))
            indices = np.where(temp_blocks == 0)
            ret = []
            for i in range(9):
                ret += [UltimateTTT_Move(i, index[0], index[1], self.player_to_move)
                        for index in list(zip(indices[0], indices[1]))
                        ]
            return ret

        local_board = self.blocks[index_local_board]
        indices = np.where(local_board == 0)

        if (len(indices[0]) != 0) and (not is_occupied):
            self.free_move = False
            return [UltimateTTT_Move(index_local_board, index[0],
                                     index[1], self.player_to_move)
                    for index in list(zip(indices[0], indices[1]))
                    ]
        # chosen block is full or occupied (1 or -1)
        self.free_move = True
        ret = []
        for i in range(9):
            if (self.global_cells[i] == 0) and (not np.all(self.blocks[i] != 0)):
                indices = np.where(self.blocks[i] == 0)
                ret += [UltimateTTT_Move(i, index[0], index[1], self.player_to_move)
                        for index in list(zip(indices[0], indices[1]))
                        ]
        return ret
