# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Tuhn Poker implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""

import enum

import numpy as np

import pyspiel


class Action(enum.IntEnum):
  PASS = 1
  ACT = 2
  LEAD = 3
  FOLLOW = 4
  OFFER = 5
  ATTACK = 6

class DynamicAction:
    @staticmethod
    def follow_chain(action_board):
        """Returns a list of available actions based on the action board."""
        return [i+7 for i, chain in enumerate(action_board)]
      
_NUM_PLAYERS = 10
# _DECK = frozenset([0, 1, 2])
_DECK =set(range(0, 10))
_GAME_TYPE = pyspiel.GameType(
    short_name="python_leviathan",
    long_name="Python Leviathan",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=5,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(Action),
    max_chance_outcomes=len(_DECK),
    num_players=_NUM_PLAYERS,
    min_utility=-2.0,
    max_utility=2.0,
    utility_sum=None,
    max_game_length=10)


class LevithanGame(pyspiel.Game):
  """A Python version of Levithan."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return LevithanState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""

    return LevithanObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True),
            params)
    
  def num_distinct_actions(self):
    return len(Action)+_NUM_PLAYERS
        
    # return LevithanObserver(
    #     iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True),
    #     params)


class LevithanState(pyspiel.State):
  """A python version of the Levithan state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.cards = []
    self.action_board = []
    self.stage = 1
    # self.pot = [2.0, 1.0]
    self._game_over = False
    self._next_player = 0
    self._NUM_PLAYERS = _NUM_PLAYERS
    self.round = 0
    self.current_action = None

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif len(self.cards) < _NUM_PLAYERS:
      return pyspiel.PlayerId.CHANCE
    else:
      return self._next_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending cards."""
    assert player >= 0
    if self.stage == 1:
      return [Action.PASS, Action.ACT]
    elif self.stage == 2:
      return [Action.LEAD] + DynamicAction.follow_chain(self.action_board)

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    outcomes = sorted(_DECK - set(self.cards))
    p = 1.0 / len(outcomes)
    return [(o, p) for o in outcomes]

  def _check_game_over(self):
    if self.round > self._NUM_PLAYERS and self.current_player() == 1:
      self._game_over = True
  
  def _apply_action(self, action):
    """Applies the specified action to the state."""
    agent_no = self._next_player
    chain_selection = 0

    # Check if the game is at a chance node
    if self.is_chance_node():
        self.cards.append(action)
    else:
        # Check if the agent decides to take an action or pass
        if action == Action.ACT:
            # Check if the agent decides to lead or follow
            self.current_action = Action.ACT
            self.stage = 2
        elif action == Action.PASS:
            self.current_action = Action.PASS
            self.stage = 1
        elif action == Action.LEAD:
            self.action_board.append([])
            self.action_board[len(self.action_board)-1].append(agent_no)
            # self.action_board[len(self.action_board)-1].append([action_details["action_type"], agent_no, action_details["target_agent_no"]])
            self.current_action = Action.LEAD
            self.stage = 1
        elif action > 6:
            chain_selection = action-7
            self.action_board[int(chain_selection)].append(agent_no)
            # self.action_board[int(chain_selection)].append([action_details["action_type"], agent_no, action_details["target_agent_no"]])
            self.current_action = Action.FOLLOW
            self.stage = 1

        # Update the next player
        if self.current_action != Action.ACT:
          self._next_player = (self._next_player + 1) % self._NUM_PLAYERS
          self.round += 1
          self.current_action = None

        # Check if the game is over
        self._check_game_over()

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal:{action}"
    elif action == Action.PASS:
      return "Pass"
    else:
      return "ACT"

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    # pot = self.pot
    # winnings = float(min(pot))
    if not self._game_over:
      return [0.0] * self._NUM_PLAYERS
    # elif pot[0] > pot[1]:
    #   return [winnings, -winnings]
    # elif pot[0] < pot[1]:
    #   return [-winnings, winnings]
    # elif self.cards[0] > self.cards[1]:
    #   return [winnings, -winnings]
    elif len(self.action_board)==0:
      return [0] * self._NUM_PLAYERS
    else:
      reward_list = [0] * self._NUM_PLAYERS
      for idx in self.action_board[0]:
        reward_list[idx] = 1
      return reward_list
    

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return ""
  
  def last(self):
    """Returns True if the game is over."""
    return self.is_terminal()

class LevithanObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")

        # Determine which observation pieces we want to include.
        pieces = [("player", _NUM_PLAYERS, (_NUM_PLAYERS,))]
        pieces.append(("cards", len(_DECK), (len(_DECK),)))
        pieces.append(("action_board", _NUM_PLAYERS, (_NUM_PLAYERS,)))

        # Build the single flat tensor.
        total_size = sum(size for name, size, shape in pieces)
        self.tensor = np.zeros(total_size, np.float32)

        # Build the named & reshaped views of the bits of the flat tensor.
        self.dict = {}
        index = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[index:index + size].reshape(shape)
            index += size

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        self.tensor.fill(0)
        if "player" in self.dict:
            self.dict["player"][player] = 1
        if "cards" in self.dict:
            for card in state.cards:
                self.dict["cards"][card] = 1
        if "action_board" in self.dict:
            for i, chain in enumerate(state.action_board):
                for player in chain:
                    self.dict["action_board"][i] = player

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        pieces = []
        if "player" in self.dict:
            pieces.append(f"p{player}")
        if "cards" in self.dict:
            pieces.append(f"cards:{state.cards}")
        if "action_board" in self.dict:
            pieces.append(f"action_board:{state.action_board}")
        return " ".join(str(p) for p in pieces)
      
## omit for the perfect information game.

# class LevithanObserver:
#   """Observer, conforming to the PyObserver interface (see observation.py)."""

#   def __init__(self, iig_obs_type, params):
#     """Initializes an empty observation tensor."""
#     if params:
#       raise ValueError(f"Observation parameters not supported; passed {params}")

#     # Determine which observation pieces we want to include.
#     pieces = [("player", 2, (2,))]
#     if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
#       pieces.append(("private_card", 3, (3,)))
#     if iig_obs_type.public_info:
#       if iig_obs_type.perfect_recall:
#         pieces.append(("betting", 6, (3, 2)))
#       else:
#         pieces.append(("pot_contribution", 2, (2,)))

#     # Build the single flat tensor.
#     total_size = sum(size for name, size, shape in pieces)
#     self.tensor = np.zeros(total_size, np.float32)

#     # Build the named & reshaped views of the bits of the flat tensor.
#     self.dict = {}
#     index = 0
#     for name, size, shape in pieces:
#       self.dict[name] = self.tensor[index:index + size].reshape(shape)
#       index += size

#   def set_from(self, state, player):
#     """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
#     self.tensor.fill(0)
#     if "player" in self.dict:
#       self.dict["player"][player] = 1
#     if "private_card" in self.dict and len(state.cards) > player:
#       self.dict["private_card"][state.cards[player]] = 1
#     if "pot_contribution" in self.dict:
#       self.dict["pot_contribution"][:] = state.pot
#     if "betting" in self.dict:
#       for turn, action in enumerate(state.bets):
#         self.dict["betting"][turn, action] = 1

#   def string_from(self, state, player):
#     """Observation of `state` from the PoV of `player`, as a string."""
#     pieces = []
#     if "player" in self.dict:
#       pieces.append(f"p{player}")
#     if "private_card" in self.dict and len(state.cards) > player:
#       pieces.append(f"card:{state.cards[player]}")
#     if "pot_contribution" in self.dict:
#       pieces.append(f"pot[{int(state.pot[0])} {int(state.pot[1])}]")
#     if "betting" in self.dict and state.bets:
#       pieces.append("".join("pb"[b] for b in state.bets))
#     return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, LevithanGame)

# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python spiel example."""

import random
from absl import app
from absl import flags
import numpy as np

from open_spiel.python import games  # pylint: disable=unused-import
import pyspiel

from python.games.leviathan_game_v1 import Action, LevithanState

FLAGS = flags.FLAGS

# Game strings can just contain the name or the name followed by parameters
# and arguments, e.g. "breakthrough(rows=6,columns=6)"
flags.DEFINE_string("game_string", "python_leviathan", "Game string")


def main(_):
  games_list = pyspiel.registered_games()
  print("Registered games:")
  print(games_list)

  action_string = None

  print("Creating game: " + FLAGS.game_string)
  game = pyspiel.load_game(FLAGS.game_string)

  # Create the initial state
  state: LevithanState = game.new_initial_state()

  # Print the initial state
  print(str(state))

  while not state.is_terminal():
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      num_actions = len(outcomes)
      print("Chance node, got " + str(num_actions) + " outcomes")
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      print("Sampled outcome: ",
            state.action_to_string(state.current_player(), action))
      state.apply_action(action)
      print(state.cards)
    # elif state.is_simultaneous_node():
    #   # Simultaneous node: sample actions for all players.
    #   random_choice = lambda a: np.random.choice(a) if a else [0]
    #   chosen_actions = [
    #       random_choice(state.legal_actions(pid))
    #       for pid in range(game.num_players())
    #   ]
    #   print("Chosen actions: ", [
    #       state.action_to_string(pid, action)
    #       for pid, action in enumerate(chosen_actions)
    #   ])
    #   state.apply_actions(chosen_actions)
    else:
      # Decision node: sample action for the single current player
      action = random.choice(state.legal_actions(state.current_player()))
      action_string = state.action_to_string(state.current_player(), action)
      print("Player ", state.current_player(), ", randomly sampled action: ",
            action_string)
      state.apply_action(action)
      
      print(state.current_action)
      if state.current_action == Action.ACT:
        print(state.action_board)
        print(state.legal_actions(state.current_player()))
        action = random.choice(state.legal_actions(state.current_player()))
        print(action)
        action_string = state.action_to_string(state.current_player(), action)
        # print("Player ", state.current_player(), ", randomly sampled action: ", action_string)
        state.apply_action(action)
        print(state.action_board)
        
    print(str(state))

  # clear self.cards before initiate new action board

  # Game is now done. Print utilities for each player
  returns = state.returns()
  for pid in range(game.num_players()):
    print("Utility for player {} is {}".format(pid, returns[pid]))


if __name__ == "__main__":
  app.run(main)

