import numpy as np
from collections import defaultdict
from state import *
from copy import deepcopy

class MonteCarloTreeSearchNode(object):
    def __init__(self, _state: State_2, parent=None):
        self._number_of_visits = 0
        # self._result = defaultdict(int)
        self._result = 0
        self.state = _state
        self.parent = parent
        self.children = []

    @property
    def untried_actions(self):
        if not hasattr(self, '_untried_actions'):
            self._untried_actions = self.state.get_valid_moves
        return self._untried_actions

    @property
    def q(self):
        # wins = self._result[self.parent.state.previous_move.value]
        # loses = self._result[-1 * self.parent.state.previous_move.value]
        # return wins - loses
        return self._result

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self.untried_actions.pop()
        move_node = deepcopy(self)
        move_node.state.act_move(action)
        child_node = MonteCarloTreeSearchNode(move_node.state, parent=self)
        self.children.append(child_node)
        return child_node

    def is_goal_state(self):
        return self.state.game_over or len(self.state.get_valid_moves) == 0

    def rollout(self):
        current_rollout_state = deepcopy(self.state)

        while not current_rollout_state.game_over:
            possible_moves = current_rollout_state.get_valid_moves
            action = self.rollout_policy(possible_moves)
            if action is not None:
                current_rollout_state.act_move(action)
            else:
                return 0

        # print('Result', current_rollout_state.game_result(current_rollout_state.global_cells.reshape(3, 3)))
        return current_rollout_state.game_result(current_rollout_state.global_cells.reshape(3, 3))

    def rollout_policy(self, possible_moves):
        if not possible_moves:
            return None
        return possible_moves[np.random.randint(len(possible_moves))]

    def backpropagate(self, result):
        self._number_of_visits += 1
        # self._result[result] += 1
        self._result += result

        # If it not a root -> continue backpropagate
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expand(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        # print('-------')
        # for c in self.children:
        #     if c.n == 0:
        #         return self.children[np.random.randint(len(self.children))]

        choices_weight = [
            (c.q / c.n) + c_param * np.sqrt(2 * np.log(self.n) / c.n) for c in self.children
        ]

        # print('Choices:', choices_weight)

        return self.children[np.argmax(choices_weight)]

        # print('Choice: ', choices_weight)
        # arg_max = max(choices_weight[i] for i in range(len(choices_weight)))
        # res = []
        # for i in range(len(choices_weight)):
        #     if choices_weight[i] == arg_max: res.append(i)
        #
        # return self.children[np.random.randint(len(res))]
