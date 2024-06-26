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

_NUM_ACTIONS = 10
_NUM_DYNA_ACTIONS = 4
_NUM_PLAYERS = 10
class Action(enum.IntEnum):
  PASS = 1
  LEAD = 2
  FOLLOW = 3

class DynamicAction:
    @staticmethod
    def lead_chain_offer(action_board):
        return [_NUM_DYNA_ACTIONS * _NUM_PLAYERS * i + _NUM_ACTIONS for i, chain in enumerate(action_board)]
    
    @staticmethod
    def lead_chain_attack(action_board):
        return [_NUM_DYNA_ACTIONS * _NUM_PLAYERS * i + 1 + _NUM_ACTIONS for i, chain in enumerate(action_board)]
  
    @staticmethod
    def follow_chain_offer(action_board):
        """Returns a list of available actions based on the action board."""
        return [_NUM_DYNA_ACTIONS * _NUM_PLAYERS * i + 2 + _NUM_ACTIONS for i, chain in enumerate(action_board)]
      
    @staticmethod
    def follow_chain_attack(action_board):
        """Returns a list of available actions based on the action board."""
        return [_NUM_DYNA_ACTIONS * _NUM_PLAYERS * i + 3 + _NUM_ACTIONS for i, chain in enumerate(action_board)]
    
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
    self.action_board = [[]]
    # self.pot = [2.0, 1.0]
    self._game_over = False
    self._next_player = 0
    self._NUM_PLAYERS = _NUM_PLAYERS
    self.round = 0
    self.is_interaction_decision = 0
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
    return [Action.PASS] + [i for i in range(10, 31) if i % 4 == 1 or 2] + DynamicAction.follow_chain_attack(self.action_board) + DynamicAction.follow_chain_offer(self.action_board)

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
        if action == Action.PASS:
            self.current_action = Action.PASS
        else:
          chain_selection = (action - _NUM_ACTIONS) // (_NUM_DYNA_ACTIONS * _NUM_PLAYERS) ## 算回去
          decoded_index = (action - _NUM_ACTIONS) % (_NUM_DYNA_ACTIONS * _NUM_PLAYERS)
          dynamic_action_code = decoded_index % 4
          target_no = decoded_index % _NUM_PLAYERS 
          is_offer = dynamic_action_code % 2 == 0
          
          if target_no == agent_no:
            return False
          
          if dynamic_action_code < 2:
            action_type = 'lead'
          else:
            action_type = 'follow'
              
          if action_type == 'lead':
            
              if is_offer:
                  self.action_board.append([['offer', agent_no, target_no]])
              else:
                  self.action_board.append([['attack', agent_no, target_no]])
              
              self.current_action = Action.LEAD
              
          elif action_type == 'follow':
              
              if is_offer:
                  self.action_board[int(chain_selection)].append(['offer', agent_no, target_no])
              else:
                  self.action_board[int(chain_selection)].append(['attack', agent_no, target_no])

              self.current_action = Action.FOLLOW

        # Update the next player
        self._next_player = (self._next_player + 1) % self._NUM_PLAYERS
        self.round += 1
        # self.current_action = None

        # Check if the game is over
        self._check_game_over()
        
        return target_no == agent_no

  def _action_to_string(self, player, action):
    """Action -> string."""
    
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal:{action}"
    else:
      return str(action)

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

# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, LevithanGame)
