import math

import numpy as np
from envs.hex.renderer import HexRenderer
from .BasicClientActorAbs import BasicClientActorAbs


class BasicClientActor(BasicClientActorAbs):

    def __init__(self, agent_factory, IP_address=None, verbose=True):
        self.agent_factory = agent_factory
        self._agent = None
        self._prev_board = None
        self.series_id = -1
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)

    def _plot(self, framerate=30):
        HexRenderer.render(
            np.array(self._prev_board, dtype=int).reshape(6, 6),
            block=False, pause=1/framerate, close=False)

    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """
        pid, *board = state
        
        # Identify which action has been played since our last move (thanks keith for not providing this)
        diff = tuple(a - b for a, b in zip(board, self._prev_board))
        if any(diff):
            # Deduce AI action
            ai_action, ai_pid = next((i, v) for i, v in enumerate(diff) if v != 0)

            # Apply action to env and cached board
            self._prev_board[ai_action] = ai_pid
            self._plot()
            self._agent.env.move(action=ai_action, player=ai_pid)
        
        # Determine next action
        action = self._agent.get_action(state)

        # Apply action to env and cached board
        self._prev_board[action] = pid
        self._plot()
        self._agent.env.move(action=action, player=pid)
        return (action // 6, action % 6)

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return

        """
        self.series_id = series_id
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player
        # The network is only trained to recommend actions from states 
        # emerging from games where 1 is the starting player.

        # If 2 is the starting player, we transform to an equivalent (but transposed)
        # game where 1 is the starting player, swapping player ID's for 1 and 2.
        
        # As a consequence, all actions played in a transformed game must be transformed
        # back to the original representation
        
        
        self._agent = self.agent_factory(start_player)
        self._prev_board = [0 for _ in range(6**2)]
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        ##############################
        print("Game over, these are the stats:")
        print('Winner: ' + str(winner))
        print('End state: ' + str(end_state))

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Series ended, these are the stats:")
        print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))


if __name__ == '__main__':
    bsa = BasicClientActor(verbose=True)
    bsa.connect_to_server()
