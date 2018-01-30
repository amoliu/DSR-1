from enum import Enum
class Optimizers(Enum):
    RMS = 0
    ADAM = 1
    ADAGRAD = 2
    MOMENTUM = 3

#Definiton of constants used during the training process

PLAY_STEPS = 4 #number of steps played in a sequence without training
MAXIMUM_PLAY_STEPS = 500000 #the maximum number of steps to play
ENV_NAME = 'PongNoFrameskip-v4' #the name of the openAI gym environment
REPLAY_BUFFER_SIZE = 1000000 #The size of the replay buffer
TRAINING_START = 50000 #30000 #Number of steps from the environment sampled before training starts
EPSILON_START = 1.0 #The epsilon exploration rate to start with
EPSILON_END = 0.1 #0.02 #The epsilon exploration rate to end with
EPSILON_END_T = 1000000 #The number of steps until the finally exploration rate is reached
BATCH_SIZE = 64#32 #The batch size normalized per environment step
PRIO_BATCH_SIZE = 12#6 #The batch size of the reward prio batch
LOG_DIR = 'pong_replica' #The directory where the log files should be placed
LEARNING_RATE = 1e-4 #Learning rate of the optimizer
NETWORK_SYNC_STEPS = 5000#10000 #Number of steps after which the networks are synchronized
SAVE_INTERVAL = 10000 #Number of steps until checkpoint is reached
FEATURE_DIM = 512 #Dimension of the SF representation
EVALUATION_INTERVAL = 50000 #frequency of network evalutation
OPTIMIZER = Optimizers.RMS

REWARD_TRAINING_START = 15000 #start of the reward training
REWARD_TRAINING_FREQ = 3000 #frequence of the reward training
REWARD_TRAINING_STEPS = 25000 #inital number of reward training steps
