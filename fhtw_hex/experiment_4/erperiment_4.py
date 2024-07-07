from fhtw_hex.utils import plot_learning_curve
import numpy as np


def board_size(size, down_sampling):
    board = [[0 for x in range(size)] for y in range(size)]
    if down_sampling in [3, 4, 5, 6]:
        # Anzahl der Reihen für Spieler Schwarz
        black_rows = {3: 4, 4: 3, 5: 2, 6: 1}[down_sampling]
        # Startreihe für Spieler Weiß
        white_start_row = {3: 4, 4: 3, 5: 2, 6: 1}[down_sampling]
        # Startspalte für Spieler Weiß
        white_start_col = {3: 3, 4: 4, 5: 5, 6: 6}[down_sampling]
        # Setze die Steine für Spieler Schwarz (-1)
        for i in range(black_rows):
            for j in range(size):
                board[i][j] = -1
        # Setze die Steine für Spieler Weiß (1)
        for i in range(white_start_row, 7):
            for j in range(white_start_col, 7):
                board[i][j] = 1
    return board


def train_vs_random(game, agent, n_games, N, figure_file):
    best_score = -np.inf
    score_history = []
    avg_score = 0
    n_steps = 0
    games_played = 0
    down_sampling = 3
    print_board = True

    for i in range(n_games):
        game.reset()
        games_played += 1
        game.board = board_size(7, down_sampling)  # Verwenden einer Kopie des initialen Boards
        
        if print_board and down_sampling <=7:
            print("--------------------------------------------------------------------------------------------------")
            print(f"Boardsize ist derzeit {down_sampling}x{down_sampling}!")
            print("Neues Board:")
            print("")
            for row in game.board:
                print(row)
            print("--------------------------------------------------------------------------------------------------")
            print_board = False

        # Anpassung des Boards nach 500 Spielen pro Spielbrettgröße
        if games_played == 250 and down_sampling <=7:
            games_played = 0
            down_sampling += 1
            print_board = True


        done = False
        score = 0
        legal_moves_per_game = 0
        illegal_moves_per_game = 0
        learn_iters = 0

        while not done:
            state = np.array(game.board).flatten()
            reward = 0

            if game.player == 1:
                action, prob, val = agent.choose_action(state)
                action_coordinates = (action // 7, action % 7)

                if action_coordinates in game.get_action_space():
                    legal_moves_per_game += 1
                    game.move(action_coordinates)
                else:
                    illegal_moves_per_game += 1
                    game.player *= -1

                if game.winner != 0:
                    print('Game winner:', game.winner)
                    if game.winner == 1:
                        reward = 1
                    else:
                        reward = 0
                    done = True

                n_steps += 1
                agent.remember(state, action, prob, val, reward, done)

            else:
                game._random_move()
                if game.winner != 0:
                    print('Game winner:', game.winner)
                    reward = 0
                    done = True

            score += reward
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

        # agent.learn()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if down_sampling >=7:
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

        print(
            f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Best Score: {best_score:.2f},'
            f' Time Steps: {n_steps}, Legal moves per game {legal_moves_per_game}, '
            f'Illegal moves per game {illegal_moves_per_game} , Learning Steps per Game: {learn_iters}')

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    print("Training completed.")


def train_agent_vs_agent(game, agent, agent_2, n_games, N, figure_file):
    best_score = -np.inf
    score_history = []
    avg_score = 0
    n_steps = 0
    games_played = 0
    down_sampling = 3
    print_board = True

    for i in range(n_games):
        game.reset()
        games_played += 1
        game.board = board_size(7, down_sampling)  # Verwenden einer Kopie des initialen Boards
        
        if print_board and down_sampling <=7:
            print("--------------------------------------------------------------------------------------------------")
            print(f"Boardsize ist derzeit {down_sampling}x{down_sampling}!")
            print("Neues Board:")
            print("")
            for row in game.board:
                print(row)
            print("--------------------------------------------------------------------------------------------------")
            print_board = False

        # Anpassung des Boards nach 500 Spielen pro Spielbrettgröße
        if games_played == 250 and down_sampling <=7:
            games_played = 0
            down_sampling += 1
            print_board = True


        done = False
        score = 0
        legal_moves_per_game = 0
        illegal_moves_per_game = 0
        learn_iters = 0

        while not done:
            state = np.array(game.board).flatten()
            reward = 0

            if game.player == 1:
                action, prob, val = agent.choose_action(state)
                action_coordinates = (action // 7, action % 7)

                if action_coordinates in game.get_action_space():
                    legal_moves_per_game += 1
                    game.move(action_coordinates)
                else:
                    illegal_moves_per_game += 1
                    game.player *= -1

                if game.winner != 0:
                    print('Game winner:', game.winner)
                    if game.winner == 1:
                        reward = 1
                    else:
                        reward = 0
                    done = True

                n_steps += 1
                agent.remember(state, action, prob, val, reward, done)

            else:
                action, _, _ = agent_2.choose_action(state)
                action_coordinates = (action // 7, action % 7)

                if action_coordinates in game.get_action_space():
                    game.move(action_coordinates)
                if game.winner != 0:
                    print('Game winner:', game.winner)
                    reward = 0
                    done = True

            score += reward
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if down_sampling >=7:
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

        print(
            f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Best Score: {best_score:.2f},'
            f' Time Steps: {n_steps}, Legal moves per game {legal_moves_per_game}, '
            f'Illegal moves per game {illegal_moves_per_game} , Learning Steps per Game: {learn_iters}')

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    print("Training completed.")