import pyspiel
import numpy as np

class LeviathanGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(pyspiel.GameType(
            short_name="python_leviathan_gpt",
            long_name="Python Leviathan GPT",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
            information=pyspiel.GameType.Information.PERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.GENERAL_SUM,
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            max_num_players=4,
            min_num_players=2,
            provides_information_state_string=True,
            provides_information_state_tensor=True,
            provides_observation_string=True,
            provides_observation_tensor=True,
        ), params)

    def new_initial_state(self):
        return LeviathanGameState(self)

    def num_distinct_actions(self):
        return 4  # attack, offer

    def num_players(self):
        return 4

    def max_game_length(self):
        return 100  # Arbitrary limit for game length

class LeviathanGameState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self.current_player = 0
        self.round_number = 0
        self.is_terminal = False
        self.history = []

    def current_player(self):
        return pyspiel.PlayerId.TERMINAL if self.is_terminal else self.current_player

    def legal_actions(self, player):
        return [0, 1]  # attack, offer

    def apply_action(self, action):
        # Implement action logic here
        self.history.append(action)
        self.current_player = (self.current_player + 1) % self.num_players()
        if self.current_player == 0:
            self.round_number += 1
            # Check if game is terminal
            if self.round_number >= self.game.max_game_length():
                self.is_terminal = True

    def is_terminal(self):
        return self.is_terminal

    def returns(self):
        return [0] * self.num_players()  # Placeholder for returns

    def __str__(self):
        return f"Round {self.round_number}, Player {self.current_player}, History: {self.history}"

    def observation_string(self, player):
        return self.__str__()

    def information_state_string(self, player):
        return self.__str__()

# Register the game with OpenSpiel
pyspiel.register_game(_GAME_TYPE, LeviathanGame)