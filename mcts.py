# - *- coding: utf-8 - *-
import numpy as np
import copy
from configs import MCTSConfig


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self.children = {}  # a map from action to TreeNode
        self.n_visits = 0
        self._W = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self.n_visits += 1
        self._W += leaf_value
        # Update Q, a running average of values for all visits.
        # self._Q += 1.0 * (leaf_value - self._Q) / self.n_visits
        self._Q = 1.0 * self._W / self.n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent.n_visits) / (1 + self.n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self.children == {}

    def is_root(self):
        return self._parent is None


class MCTS:
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn):
        """
        :param policy_value_fn: a function takes in a board state and
        outputs a list of (action, probability) tuples and also a score in [
        -1, 1] (i.e. the expected value of the end game score from the
        current player's perspective) for the current player.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._mcts_config = MCTSConfig()
        self._c_puct = self._mcts_config.c_put
        self._n_playout = self._mcts_config.n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        # if "true" leaf_value is -1 (current player loses), the node
        # should have value 1 cause the do_move flip the current_player
        # which means node represents opponent's state
        # From my perspective, the "true" leaf_value can only be -1 cause if
        # winner is existing, it must be previous player(do_move flip player)
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.n_visits)
                      for act, node in self._root.children.items()]
        acts, visits = zip(*act_visits)
        temp = self._mcts_config.temperature
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root.children:
            self._root = self._root.children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


class MCTSPlayer:
    """AI player based on MCTS"""

    def __init__(self, policy_value_fn, is_selfplay=False):
        self.mcts = MCTS(policy_value_fn)
        self._is_selfplay = is_selfplay

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, is_return_prob=False):
        sensible_moves = board.available_positions
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                        acts,
                        # Changing the third parameter makes it searches deeper
                        p=0.75 * probs + 0.25 * np.random.dirichlet(
                                0.3 * np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
            #   location = board.move_to_location(move)
            #   print("AI move: %d,%d\n" % (location[0], location[1]))

            if is_return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS Player"
