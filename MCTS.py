import logging
import math 
import numpy as np 


# args = dotdict({
#     'numIters': 1000,
#     'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
#     'tempThreshold': 15,        #
#     'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
#     'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
#     'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
#     'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
#     'cpuct': 1,

#     'checkpoint': './temp/',
#     'load_model': False,
#     'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
#     'numItersForTrainExamplesHistory': 20,

# })

EPS = 1e-8 

log = logging.getLogger(__name__)

class MCTS():
    """
    This class handles the MCTS
    """
    
    def __init__(self, game, nnet, args):
        
        self.game = game 
        self.nnet = nnet
        self.args = args
        
        self.Qsa = {} # stores Q values for s,a (as defined in the paper)
        self.Nsa = {} # stores #times edge s,a was visited 
        self.Ns = {}  # stores #times board s was visited 
        self.Ps = {}  # stores initial policy (returned by neural net)
        
        self.Es = {} # stores game.getGameEnded ended for board s
        self.Vs = {} # stores game.getValideMoves for board s
        
    def getActionProb(self, canonicalBoard, temp = 1):
        """
        This function performs numMCTSSims simulationns of MCTS starting from 
        canonicalBoard.
        
        returns:
            probs: a policy vector where the probability of the ith action 
            is proportional to Nsa[(s,a)] ** (1./temp)
            
            based on the article: pi(a|s0) = N(s0,a)**1/temp / sum(N(s0,i)**1/temp)
            
        """
        
        for i in range(self.args.numMCTSims):
            self.search(canonicalBoard)
            
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        
        
        if temp == 0: # greedy algorithm
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs 
        
        counts = [x ** (1./ temp) for x in counts]
        counts_sum = float(sum(counts))
        
        probs = [x / counts_sum for x in counts]
        
        return probs
    
    
    def search(self,canonicalBoard):
        
        """
        This function performes one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that has 
        the maximum upper confidence bound as in the paper.
        
        Once a leaf node is found, the neural network is called to return an 
        initial policy P and state value v for the state. This value is propagated up 
        the search path. In case the leaf node is a terminal state, the outcome is propagated
        up the search path. The values of Ns, Nsa, Qsa are updated.
        
        Note: the return values are the negative of the value of the current state.
        This is done since v is in [-1,1] and if v is the value of a state for the current palyer,
        then its value is -v for the other player. 
        
        Returns: 
            v: the negative of teh value of the current canonicalBoard
        """
        
        s = self.game.stringRepresentation(canonicalBoard)
        
        if s not in self.Es: # (means the state is not terminal state)
            self.Es[s] = self.game.getGameEnded(canonicalBoard,1) # return 1 if player won, -1 loss, 0 if the game not ended
            # and small value if it is a draw
        if self.Es[s] != 0 :
            # terminal node 
            return -self.Es[s]
        
        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard,1)
            self.Ps[s] = self.Ps[s] * valids # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            
            if sum_Ps_s > 0 :
                self.Ps[s] = self.Ps[s] / sum_Ps_s #renomalization
                
            else:
                # if all valid moves were masked, make all valid moves equally probable
                
                # NB! All valid moves may by masked if either your NNet architecture is insufficient
                # or you have get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your 
                # NNet  and/or training process.
                
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] = self.Ps[s] / np.sum(self.Ps[s])
            
            self.Vs[s] = valids
            self.Ns[s] = 0
            
            return -v 
        
        valids = self.Vs[s]
        cur_best = - float('inf')
        best_act = - 1
        
        # Pick the action with the heigest UCB
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                        1+self.Nsa[(s,a)])
                    
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)/(1+0) # Q = 0
                    
                
                if u > cur_best:
                    cur_best = u 
                    best_act = a
        
        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard,1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        
        v = self.search(next_s)
        
        if (s,a) in self.Qsa:
             W = (self.Nsa[(s,a)] * self.Qsa[(s,a)] + v)
             self.Qsa[(s,a)] = W/(self.Nsa[(s,a)]+1) 
             self.Nsa[(s,a)] += 1
             
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1
            
        self.Ns[s] += 1
        
        return -v
             
        
                    
                    
                        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            