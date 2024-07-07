from fhtw_hex.utils import plot_learning_curve, plot_win_history
import numpy as np
from fhtw_hex.reward_utils import is_adjecent_move, is_blocking_move, is_connection_blocking_move, is_connecting_move


def train_vs_random(game, agent, agent_player, n_games, N, figure_file, phase):
    best_score = -np.inf
    score_history = []
    win_history = []
    win_count_agent = 0
    n_steps = 0
    for i in range(n_games):
        game.reset()
        done = False
        score = 0
        legal_moves_per_game = 0
        learn_iters = 0

        while not done:
            state = np.array(game.board).flatten()
            reward = 0

            if game.player == agent_player:
                action, prob, val = agent.choose_action(state)
                action_coordinates = (action // 7, action % 7)

                if action_coordinates in game.get_action_space():
                    legal_moves_per_game += 1
                    previous_board = np.copy(game.board)
                    # game.print()
                    game.move(action_coordinates)

                    if is_blocking_move(previous_board, action_coordinates, agent_player):
                        if game.player == 1:
                            reward += 0.5  # Reward for blocking

                    if is_connection_blocking_move(previous_board, action_coordinates, agent_player):
                        if game.player == 1:
                            # print(action_coordinates)
                            # game.print()
                            reward += 1  # Reward for blocking

                    if is_adjecent_move(previous_board, action_coordinates, agent_player):
                        if game.player == 1:
                            reward += 1 # Reward for connecting

                    if is_connecting_move(previous_board, action_coordinates, agent_player):
                        if game.player == 1:
                            reward += 4 # Reward for connecting

                if game.winner != 0:
                    win_history.append(game.winner)
                    print('Game winner:', game.winner)
                    win_count_agent += 1
                    reward += 20+5*(7/legal_moves_per_game)**2
                    done = True

                n_steps += 1
                agent.remember(state, action, prob, val, reward, done)

            else:
                game._random_move()
                if game.winner != 0:
                    win_history.append(game.winner)
                    print('Game winner:', game.winner)
                    reward = -30
                    done = True

            score += reward
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

        agent.learn()

        score_history.append(score)

        avg_win_rate_agent = win_count_agent / (i + 1)
        win_rate_last_300 = agent_player*(np.mean(win_history[-300:])+agent_player)/2
        win_rate_last_100 = agent_player*(np.mean(win_history[-100:])+agent_player)/2
        avg_score = np.mean(score_history[-100:])

        if i >= 10:
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

        print(
            f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Best Score: {best_score:.2f},'
            f' Time Steps: {n_steps}, Legal moves per game {legal_moves_per_game}, '
            f'Win rate: {avg_win_rate_agent:.2f} , Win rate 100: {win_rate_last_100:.2f} '
            f'Win rate 300: {win_rate_last_300:.2f}, Learning Steps per Game: {learn_iters}')

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, phase, figure_file)

    x = [i + 1 for i in range(len(win_history))]
    plot_win_history(x, win_history, phase, figure_file)
    print("Training completed.")





def train_agent_vs_agent(game, agent, agent_player, agent_2, n_games, N, figure_file, phase):
    best_score = -np.inf
    score_history = []
    win_history = []
    win_count_agent = 0
    n_steps = 0
    for i in range(n_games):
        game.reset()
        done = False
        score = 0
        legal_moves_per_game = 0
        learn_iters = 0

        while not done:
            state = np.array(game.board).flatten()
            reward = 0

            if game.player == agent_player:
                action, prob, val = agent.choose_action(state)
                action_coordinates = (action // 7, action % 7)

                if action_coordinates in game.get_action_space():
                    legal_moves_per_game += 1
                    previous_board = np.copy(game.board)
                    game.move(action_coordinates)

                    # if is_blocking_move(previous_board, action_coordinates, agent_player):
                    #     if game.player == 1:
                    #         reward += 0.5  # Reward for blocking
                    #
                    # if is_connection_blocking_move(previous_board, action_coordinates, agent_player):
                    #     if game.player == 1:
                    #         # print(action_coordinates)
                    #         # game.print()
                    #         reward += 1  # Reward for blocking
                    #
                    # if is_adjecent_move(previous_board, action_coordinates, agent_player):
                    #     if game.player == 1:
                    #         reward += 1 # Reward for connecting
                    #
                    # if is_connecting_move(previous_board, action_coordinates, agent_player):
                    #     if game.player == 1:
                    #         reward += 2 # Reward for connecting

                if game.winner != 0:
                    win_history.append(game.winner)
                    print('Game winner:', game.winner)
                    win_count_agent += 1
                    reward += 20+5*(7/legal_moves_per_game)**2
                    done = True

                n_steps += 1
                agent.remember(state, action, prob, val, reward, done)

            else:
                flipped_board = game.recode_black_as_white()
                state = np.array(flipped_board).flatten()
                action, _, _ = agent_2.choose_action(state)
                action_coordinates_flipped = (action // 7, action % 7)
                action_coordinates = game.recode_coordinates(action_coordinates_flipped)
                if action_coordinates in game.get_action_space():
                    game.move(action_coordinates)
                else:
                    print("not available")

                if game.winner != 0:
                    win_history.append(game.winner)
                    print('Game winner:', game.winner)
                    reward = -30
                    done = True

            score += reward
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

        agent.learn()

        score_history.append(score)

        avg_win_rate_agent = win_count_agent / (i + 1)
        win_rate_last_300 = agent_player*(np.mean(win_history[-300:])+agent_player)/2
        win_rate_last_100 = agent_player*(np.mean(win_history[-100:])+agent_player)/2
        avg_score = np.mean(score_history[-100:])

        if i >= 10:
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

        print(
            f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Best Score: {best_score:.2f},'
            f' Time Steps: {n_steps}, Legal moves per game {legal_moves_per_game}, '
            f'Win rate: {avg_win_rate_agent:.2f} , Win rate 100: {win_rate_last_100:.2f} '
            f'Win rate 300: {win_rate_last_300:.2f}, Learning Steps per Game: {learn_iters}')

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, phase, figure_file)

    x = [i + 1 for i in range(len(win_history))]
    plot_win_history(x, win_history, phase, figure_file)
    print("Training completed.")
