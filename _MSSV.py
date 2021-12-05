import numpy as np
from mcts.node import *
from mcts.search import MonteCarloTreeSearch

def select_move(cur_state, remain_time):

    # if len(cur_state.get_valid_moves) == 81:
    #     return UltimateTTT_Move(4, 1, 1, 1)

    root = MonteCarloTreeSearchNode(cur_state, parent=None)
    mcts = MonteCarloTreeSearch(root)
    best_node = mcts.best_action(20)

    return best_node.state.previous_move
