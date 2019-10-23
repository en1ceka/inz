import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import gc
import matplotlib.cm as cm
from testing import pgame
import _thread

def convert(board):
    brd = []
    for row in board:
        for index in row:
            if index == 1:
                brd.append(1)
            else:
                brd.append(0)
    for row in board:
        for index in row:
            if index == 2:
                brd.append(1)
            else:
                brd.append(0)
    for row in board:
        for index in row:
            if index == -1:
                brd.append(1)
            else:
                brd.append(0)
    for row in board:
        for index in row:
            if index == -2:
                brd.append(1)
            else:
                brd.append(0)
    res_brd = np.resize(np.asarray(brd),(32,8))
    return res_brd

class Game:
    def __init__(self, agent1, agent2):
        self.board = np.array([[0,-1,0,-1,0,-1,0,-1],
                                [-1,0,-1,0,-1,0,-1,0],
                                [0,-1,0,-1,0,-1,0,-1],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0],
                                [1,0,1,0,1,0,1,0],
                                [0,1,0,1,0,1,0,1],
                                [1,0,1,0,1,0,1,0]])

        self.mygame = pgame(self.board)
        _thread.start_new_thread(self.mygame.draw_board, ())
        self.black_turn = True
        self.game_over = False
        self.black_won = False
        self.white_won = False
        self.draw = False
        self.count = 0
        self.agent = [agent1, agent2]

        self.black_train_x = np.empty(shape=[0, 32, 8])
        self.black_train_y = np.empty(shape=[0, 1])
        self.white_train_x = np.empty(shape=[0, 32, 8])
        self.white_train_y = np.empty(shape=[0, 1])
    
    def reset(self):
        self.board = np.array([[0,-1,0,-1,0,-1,0,-1],
                                [-1,0,-1,0,-1,0,-1,0],
                                [0,-1,0,-1,0,-1,0,-1],
                                [0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0],
                                [1,0,1,0,1,0,1,0],
                                [0,1,0,1,0,1,0,1],
                                [1,0,1,0,1,0,1,0]])
        self.black_turn = True
        self.game_over = False
        self.black_won = False
        self.white_won = False
        self.draw = False
        self.count = 0
        
    def valid_moves(self):
        valid_moves = []
        c = -1 if self.black_turn else 1
        for y in range(8):
            for x in range(8):
                if self.board[y][x] == 1*c or self.board[y][x] == 2*c:
                    mr = 2 if self.board[y][x] == 2*c else 1
                    for _ in range(mr):
                        if mr == 2:
                            c *= -1
                            if self.black_turn:
                                g = -1*c
                            else:
                                g = c
                        else:
                            g = 1
                        if y-1*c >= 0 and y-1*c < 8 and x-1 >= 0 and x-1 < 8:
                            if self.board[y-1*c][x-1] == 0:
                                valid_moves.append({'from': [y,x], 'to': [y-1*c, x-1]})
                        if y-1*c >= 0 and y-1*c < 8 and x+1 >= 0 and x+1 < 8:
                            if self.board[y-1*c][x+1] == 0:
                                valid_moves.append({'from': [y,x], 'to': [y-1*c, x+1]})
                        if y-1*c >= 0 and y-1*c < 8 and x-1 >= 0 and x-1 < 8 and y-2*c >= 0 and y-2*c < 8 and x-2 >= 0 and x-2 < 8:
                            if (self.board[y-1*c][x-1] == -1*c*g or self.board[y-1*c][x-1] == -2*c*g) and self.board[y-2*c][x-2] == 0:
                                valid_moves.append({'from': [y,x], 'to': [y-2*c, x-2], 'del': [y-1*c, x-1]})
                        if y-1*c >= 0 and y-1*c < 8 and x+1 >= 0 and x+1 < 8 and y-2*c >= 0 and y-2*c < 8 and x+2 >= 0 and x+2 < 8:
                            if (self.board[y-1*c][x+1] == -1*c*g or self.board[y-1*c][x+1] == -2*c*g) and self.board[y-2*c][x+2] == 0:
                                valid_moves.append({'from': [y,x], 'to': [y-2*c, x+2], 'del': [y-1*c, x+1]})
               
        if not valid_moves:
            if self.black_turn:
                self.white_won = True 
            else:
                self.black_won = True
            self.game_over = True

        if self.count == 60:
            self.draw = True
            self.game_over = True

        jump_moves = []
        for move in valid_moves:
            if 'del' in move:
                jump_moves.append(move)

        if jump_moves:
            return jump_moves

        return valid_moves

    def possible_boards(self):
        boards = []
        for move in self.valid_moves():
            temp_board = self.board.copy()
            temp_board[move['to'][0]][move['to'][1]] = temp_board[move['from'][0]][move['from'][1]]
            if (move['to'][0] == 7 and self.black_turn and temp_board[move['to'][0]][move['to'][1]] == -1) or (move['to'][0] == 0 and not self.black_turn and temp_board[move['to'][0]][move['to'][1]] == 1):
                temp_board[move['to'][0]][move['to'][1]] *= 2
            temp_board[move['from'][0]][move['from'][1]] = 0
            if 'del' in move:
                temp_board[move['del'][0]][move['del'][1]] = 0
            boards.append(temp_board)

        return boards
        
    def make_move(self, move):
        self.count += 1
        self.board[move['to'][0]][move['to'][1]] = self.board[move['from'][0]][move['from'][1]]
        if (move['to'][0] == 7 and self.black_turn and self.board[move['to'][0]][move['to'][1]] == -1) or (move['to'][0] == 0 and not self.black_turn and self.board[move['to'][0]][move['to'][1]] == 1):
            self.board[move['to'][0]][move['to'][1]] *= 2
        self.board[move['from'][0]][move['from'][1]] = 0
        if 'del' in move:
            self.count = 0
            self.board[move['del'][0]][move['del'][1]] = 0
        
        res_brd = convert(self.board)

        if self.black_turn:
            self.black_train_x = np.append(self.black_train_x, [res_brd], axis=0)
        else:
            self.white_train_x = np.append(self.white_train_x, [res_brd], axis=0)

        self.black_turn = not self.black_turn
    
    def play_game(self, x):
        for i in range(1):
            
            black_w = 0
            white_w = 0
            for cc in range(x):
                counter = 0
                while not self.game_over:
                    if self.valid_moves():
                        predicted_moves = self.agent[counter%2].predict(self.possible_boards())
                        max_value = [predicted_moves[0][1], 0]
                        for i, move in enumerate(predicted_moves):
                            if move[1] > max_value[0]:
                                max_value[0] = move[1]
                                max_value[1] = i
                        # print(max_value)
                        # print(counter)
                        val_moves = self.valid_moves()
                        rand = random.randint(1,101)
                    
                        if rand > 20:
                            self.make_move(val_moves[max_value[1]])
                        else:
                            # print("random")
                            self.make_move(random.choice(val_moves))
                    
                    self.mygame.set_board(self.board)
                    # plt.imshow(self.board)
                    # plt.show()
                    counter += 1
                
                if self.black_won:
                    black_w += 1
                    print('black won')
                    arr1 = np.ones(((len(self.black_train_x) - len(self.black_train_y)),1))
                    self.black_train_y = np.append(self.black_train_y, arr1, axis=0)

                    arr2 = np.zeros(((len(self.white_train_x) - len(self.white_train_y)),1))
                    self.white_train_y = np.append(self.white_train_y, arr2, axis=0)
                elif self.white_won:
                    white_w += 1
                    print('white won')
                    arr1 = np.zeros(((len(self.black_train_x) - len(self.black_train_y)),1))
                    self.black_train_y = np.append(self.black_train_y, arr1, axis=0)

                    arr2 = np.ones(((len(self.white_train_x) - len(self.white_train_y)),1))
                    self.white_train_y = np.append(self.white_train_y, arr2, axis=0)
                elif self.draw:
                    print('draw')
                    arr1 = np.zeros(((len(self.black_train_x) - len(self.black_train_y)),1))
                    self.black_train_y = np.append(self.black_train_y, arr1, axis=0)

                    arr2 = np.zeros(((len(self.white_train_x) - len(self.white_train_y)),1))
                    self.white_train_y = np.append(self.white_train_y, arr2, axis=0)
                
                print(counter)
                print('...', cc)
                
                # print(self.black_train_x.shape)
                # print(self.black_train_y.shape)
                # print(self.white_train_x.shape)
                # print(self.white_train_y.shape)
                self.reset()
            
            print('black:', black_w,'white:', white_w)
            self.agent[0].train(self.black_train_x, self.black_train_y)
            self.agent[1].train(self.white_train_x, self.white_train_y)
            self.agent[0].save_weights()
            self.agent[1].save_weights()

            self.black_train_x = np.empty(shape=[0, 32, 8])
            self.black_train_y = np.empty(shape=[0, 1])
            self.white_train_x = np.empty(shape=[0, 32, 8])
            self.white_train_y = np.empty(shape=[0, 1])

class Agent:

    def __init__(self, name):
        self.model = self.new_model()
        self.name = name

    def load_weights(self):
        print("loading weights.....")
        try:
            self.model.load_weights(self.name)
        except Exception as e:
            print(e)
        print("end")

    def save_weights(self):
        print("saving weights.....")
        self.model.save_weights(self.name)
        
        print("end")

    def new_model(self):
        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(32, 8)),
          tf.keras.layers.Dense(512, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(128, activation="relu"),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(2, activation='softmax')
          ])

        model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        return model

    def predict(self, boards):
        # tf.keras.backend.clear_session()
        # gc.collect()
        # self.load_weights()
        pred_boards = np.empty(shape=[0, 32, 8])
        for board in boards:
            pred_boards = np.append(pred_boards,[convert(board)], axis=0)

        predictions = self.model.predict(pred_boards)
        return predictions

    def train(self, x_t, y_t):
        zeros = 0
        ones = 0
        X = x_t
        y = y_t
        for row in y:
            if row[0] == 0.0:
                zeros += 1
            else:
                ones += 1

        print('zero', zeros, 'ones', ones)

        to_del = abs(zeros - ones) 

        if zeros > ones:
            for i in reversed(range(len(y))):
                if y[i][0] == 0.0:
                    y = np.delete(y, i, axis=0)
                    X = np.delete(X, i, axis=0)
                    to_del -= 1
                    if to_del == 0:
                        break
        if ones > zeros:
            for i in reversed(range(len(y))):
                if y[i][0] == 1.0:
                    y = np.delete(y, i, axis=0)
                    X = np.delete(X, i, axis=0)
                    to_del -= 1
                    if to_del == 0:
                        break

        zeros = 0
        ones = 0               
        for row in y:
            if row[0] == 0.0:
                zeros += 1
            else:
                ones += 1
         
        print('zero', zeros, 'ones', ones)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
            self.model.fit(X_train, y_train, epochs=1, batch_size=16, validation_data=(X_test, y_test))
        except Exception as e:
            print(e)

for _ in range(2):
    a1 = Agent('m1d')
    a2 = Agent('m2drand')
    a1.load_weights()
    a2.load_weights()
    g = Game(a1,a2)


    g.play_game(50)
    del a1
    del a2
    del g
    gc.collect()