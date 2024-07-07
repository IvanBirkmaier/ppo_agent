from fhtw_hex.utils import plot_learning_curve
import numpy as np
from fhtw_hex.reward_utils import is_connection_blocking_move, is_connecting_move, can_win_next_move, \
    will_opponent_win_next


def train(game, agent1, agent2, n_games, N, figure_file, seed=42):

    win_count_agent1 = 0
    win_count_agent2 = 0
    score_history_agent1 = []
    score_history_agent2 = []
    n_steps = 0

    for i in range(n_games):
        game.reset()
        done = False
        reward_agent1 = 0
        reward_agent2 = 0
        learn_iters_agent1 = 0
        learn_iters_agent2 = 0
        moves_agent1 = 0
        moves_agent2 = 0

        starting_player = 1 if (i // 100) % 2 == 0 else -1 #switch side
        game.player = starting_player
        print(f'Game {i}, Starting player: {"Agent1" if starting_player == 1 else "Agent2"}')

        while not done:
            state = np.array(game.board).flatten()
            current_agent = agent1 if game.player == 1 else agent2
            opponent_agent = agent2 if game.player == 1 else agent1

            board_size = len(game.board)
            action, prob, val = current_agent.choose_action(state)
            action_coordinates = (action // board_size, action % board_size)

            if action_coordinates in game.get_action_space():
                previous_board = [row[:] for row in game.board]

                blocking_coordinates = will_opponent_win_next(game,
                                                              game.player)  # soll verhindern, dass der gegner mit dem nächsten zug gewinnt
                if blocking_coordinates and action_coordinates == blocking_coordinates:
                    if game.player == 1:
                        reward_agent1 += 3.0
                    else:
                        reward_agent2 += 3.0

                if is_connection_blocking_move(previous_board, action_coordinates,
                                               game.player):  # agent soll generell blockieren

                    if game.player == 1:
                        reward_agent1 += 2.5
                    else:
                        reward_agent2 += 2.5
                else:
                    if game.player == 1:
                        reward_agent1 -= 0.75
                    else:
                        reward_agent2 -= 0.75

                if is_connecting_move(previous_board, action_coordinates,
                                      game.player):  # agent soll auch lernen, seine steine zu verbinden
                    if game.player == 1:
                        reward_agent1 += 2.5
                    else:
                        reward_agent2 += 2.5
                else:
                    if game.player == 1:
                        reward_agent1 -= 0.75
                    else:
                        reward_agent2 -= 0.75

                if can_win_next_move(game, action_coordinates,
                                     game.player):  # agent soll erkennen, dass er mit dem nächsten Zug gewinnt
                    if game.player == 1:
                        reward_agent1 += 3
                    else:
                        reward_agent2 += 3

                game.move(action_coordinates)

                if current_agent == agent1:
                    moves_agent1 += 1
                else:
                    moves_agent2 += 1

            if game.winner != 0:
                if game.winner == 1:
                    win_count_agent1 += 1
                    winner_agent = 'Agent1'
                else:
                    win_count_agent2 += 1
                    winner_agent = 'Agent2'

                reward_agent1 += 10 if game.winner == 1 else -10 # Belohung fürs Gewinnen und Bestrafung bei Verlieren
                reward_agent2 += 10 if game.winner == -1 else -10

                reward_agent1 -= moves_agent1 * 0.01  # Belohnung für weniger Züge
                reward_agent2 -= moves_agent2 * 0.01

                done = True
                print(f'Game winner: {winner_agent}')

            n_steps += 1
            current_agent.remember(state, action, prob, val,
                                   reward_agent1 if current_agent == agent1 else reward_agent2, done)
            opponent_agent.remember(state, action, prob, val,
                                    reward_agent2 if current_agent == agent1 else reward_agent1, done)

            if n_steps % N == 0:
                agent1.learn()
                agent2.learn()
                learn_iters_agent1 += 1
                learn_iters_agent2 += 1

        score_history_agent1.append(reward_agent1)
        score_history_agent2.append(reward_agent2)
        avg_win_rate_agent1 = win_count_agent1 / (i + 1)
        avg_win_rate_agent2 = win_count_agent2 / (i + 1)

        if i + 1 == n_games:
            agent1.save_models()
            agent2.save_models()

        print(
            f'Episode {i}, Agent1 Score: {reward_agent1:.1f}, Win Rate Agent1: {avg_win_rate_agent1:.2f}, '
            f'Agent2 Score: {reward_agent2:.1f}, Win Rate Agent2: {avg_win_rate_agent2:.2f},\n '
            f'Time Steps: {n_steps}, Moves per Game: Agent1: {moves_agent1}, Agent2: {moves_agent2}, '
            f'Learning Steps per Game: Agent1: {learn_iters_agent1}, Agent2: {learn_iters_agent2} '
        )

    x = [i + 1 for i in range(len(score_history_agent1))]
    plot_learning_curve(x, score_history_agent1, figure_file + '_agent1.png')
    plot_learning_curve(x, score_history_agent2, figure_file + '_agent2.png')
    print("Training completed.")

    return agent1, agent2
