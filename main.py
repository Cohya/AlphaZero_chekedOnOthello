
import logging
import coloredlogs
import sys
sys.path.append('./Othello')
sys.path.append('./Neural_nets')
from Coach import Coach
from OthelloGame import OthelloGame as Game 
from NNet import NNetWrapper as nn 
from utiles import dotdict

log = logging.getLogger(__name__)

coloredlogs.install(level = 'INFO') #Change this to DEBUG to see more info.
#100
args = dotdict({
        'numIters': 1000,
        'numEps' : 2,             #(was 100) Number of comlete self-play games to simulate during a new iteration
        'tempThreshold': 15,        #
        'updateThreshold': 0.6,     # During arena playoff, new nn will be accepted if threshold or more of games are won.)
        'maxlenOfQueue': 200000,    # Number of game examples to train nn's.
        'numMCTSims': 25,          # Number of games move for MCTS to simulate.
        'arenaCompare': 4,         # was (40)Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,
        'checkpoint': './checkpoint_and_weights',
        'load_model': False,
        'load_folder_file': ('folder', 'best...'),
        'numItersForTrainExamplesHistory': 20,
        })

# def main():
log.info('Loading %s...', Game.__name__)
g = Game(6)

log.info('Loading %s...', nn.__name__)
nnet = nn(g)

if args.load_model:
    log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
    nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
else:
    log.warning('Not loading a checkpoint!')
    
log.info('Loading the Coach...')
c = Coach(g, nnet, args)

if args.load_model:
    log.info("Loading 'trainExamples' from file...")
    c.loadTrainExamples()

log.info('Starting the learning process!')
c.learn()


# if __name__ == "__main__":
#     main()