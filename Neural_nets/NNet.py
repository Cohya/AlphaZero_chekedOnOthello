
import tensorflow as tf 
import numpy as np 

from OthelloNNet import OthelloNNet 
from utiles import dotdict
import time 

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
}) 


class NNetWrapper():
    def __init__(self, game):
        self.nnet = OthelloNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        
    def train(self, examples):
        """
        examples: list of examples, each example is form (board, pi, vi)

        """
        input_boards, target_policy, target_value = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_policy = np.asarray(target_policy)
        target_value = np.asarray(target_value)
        # print(input_boards.shape)
        self.nnet.fit(X = input_boards, Y = [target_policy, target_value],
                            batch_sz = args.batch_size, epochs = args.epochs)# it was model.fit(for keras)
        
    def predict(self, board):
        """
        board: np array with board
        """
        # timing 
        start = time.time()
        
        # preparing input
        # print("I am in predict")
        if len(board.shape) <4:
            board = np.expand_dims(board, axis=-1)
            board = np.expand_dims(board, axis=0)
            
        # print(board.shape)
        # run 
        policy, value = self.nnet.predict(board)
        
        # print("Move time: " , time.time()-start)
        
        return policy[0], value[0]
    
    
    def save_checkpoint(self, folder = 'checkpoint', file_name = 'checkpoint.pickle'):
        self.nnet.save_weights(folder = folder, file_name = file_name)
        
    def load_checkpoint(self, folder = 'checkpoint', file_name = 'checkpoint.pickle'):
        self.nnet.load_weights(folder, file_name)
            
        
        
        
        
        
        