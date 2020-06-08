import numpy as np


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def validMoves(self, board):
        moves = []
        for col in range(7):
            for row in range(5, -1, -1):
                if board[row][col] == 0:
                    moves.append((row, col))
                    break
        return moves

    def designateOpponent(self):
        if (self.player_number == 1):
            return 2
        else:
            return 1

    def checkBoard(self, board, num, player_num):
        numZeroes = 4 - num
        player_win_str = ('{0}' * num)
        player_win_str = player_win_str.format(player_num)
        player_win_str += ('{0}' * numZeroes)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_vertical(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_vertical(board) or
                check_diagonal(board))

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        def min_value(board, alpha, beta, depth, player, opponent):
            valid = self.validMoves(board)
            if depth <= 1 or self.checkBoard(board, 4, player) or self.checkBoard(board, 4, opponent):
                return self.evaluation_function(board) * depth
            v = 1000000
            for row, col in valid:
                mockboard = np.copy(board)
                mockboard[row][col] = opponent
                result = max_value(mockboard, alpha, beta, depth - 1, player, opponent)
                v = min(v, result)
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        def max_value(board, alpha, beta, depth, player, opponent):
            valid = self.validMoves(board)
            if depth <= 1 or self.checkBoard(board, 4, player) or self.checkBoard(board, 4, opponent):
                return self.evaluation_function(board) * depth
            v = -1000000
            for row, col in valid:
                mockboard = np.copy(board)
                mockboard[row][col] = player
                result = min_value(mockboard, alpha, beta, depth - 1, player, opponent)
                v = max(v, result)
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def alpha_beta(board, depth, alpha, beta, player, opponent):
            values = []
            mockboard = np.copy(board)
            valid = self.validMoves(mockboard)
            for row, col in valid:
                mockboard[row][col] = player
                value = max(alpha, min_value(mockboard, alpha, beta, depth - 1, player, opponent))
                values.append((value, col))
            max_val = max(values, key=lambda i: i[1])[0]
            for item in values:
                if max_val in item:
                    maxindex = item[1]
                    break
            return maxindex

        opponent = self.designateOpponent()
        player = self.player_number
        return alpha_beta(board, 5, -1000000, 1000000, player, opponent)

        # raise NotImplementedError('Whoops I don\'t know what to do')

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        def expectimax(board, depth, player, opponent):
            values = []
            v = -1000000
            mockboard = np.copy(board)
            valid = self.validMoves(mockboard)
            for row, col in valid:
                mockboard[row][col] = player
                exVal = expected_value(mockboard, depth - 1, player, opponent)
                v = max(v, exVal)
                values.append((v, col))

            max_val = max(values, key=lambda i: i[1])[0]
            for item in values:
                if max_val in item:
                    max_col = item[1]
                    break
            return max_col

        def max_value(board, depth, player, opponent):
            valid_moves = self.validMoves(board)
            if depth <= 1 or self.checkBoard(board, 4, player) or self.checkBoard(board, 4, opponent):
                return self.evaluation_function(board) * depth
            v = -1000000
            for row, col in valid_moves:
                mockboard = np.copy(board)
                mockboard[row][col] = player
                exVal = expected_value(mockboard, depth - 1, player, opponent)
                v = max(v, exVal)
            return v

        def expected_value(board, depth, player, opponent):
            valid_moves = self.validMoves(board)
            lengthMoves = len(valid_moves)
            if depth <= 1 or self.checkBoard(board, 4, player) or self.checkBoard(board, 4, opponent):
                return self.evaluation_function(board) * depth
            v = 0
            for row, col in valid_moves:
                mockboard = np.copy(board)
                mockboard[row][col] = opponent
                val = max_value(mockboard, depth - 1, player, opponent)
                v += val
            return v / lengthMoves

        opponent = self.designateOpponent()
        player = self.player_number
        return (expectimax(board, 4, player, opponent))

        # raise NotImplementedError('Whoops I don\'t know what to do')

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        opponent = self.designateOpponent()
        player = self.player_number
        value = 0

        if self.checkBoard(board, 4, player):
            value += 200000
        if self.checkBoard(board, 4, opponent):
            value -= 200000
        if self.checkBoard(board, 3, player):
            value += 10000
        if self.checkBoard(board, 3, opponent):
            value -= 10000
        if self.checkBoard(board, 2, player):
            value += 2000
        if self.checkBoard(board, 2, opponent):
            value -= 2000

        return value


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move
