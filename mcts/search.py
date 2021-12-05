from mcts.node import MonteCarloTreeSearchNode
from copy import deepcopy

class MonteCarloTreeSearch:

    def __init__(self, node: MonteCarloTreeSearchNode):
        self.root = node

    def best_action(self, simulations_number):

        for i in range(0, simulations_number):
            # print('i=', i)
            v = self.tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.root.best_child()

    def tree_policy(self):
        current_node = self.root
        while not current_node.is_goal_state():
            if not current_node.is_fully_expand():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
