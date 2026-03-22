# AlphaZero hyperparameters for JungleChess

# Board
BOARD_ROWS = 9
BOARD_COLS = 7
INPUT_CHANNELS = 17       # 8 blue planes + 8 red planes + 1 player plane
ACTION_SPACE = 8686       # from_row*1000 + from_col*100 + to_row*10 + to_col

# Network
NUM_RES_BLOCKS = 10
NUM_FILTERS = 128
POLICY_FILTERS = 32
VALUE_FILTERS = 4

# MCTS
NUM_SIMULATIONS_TRAIN = 800
NUM_SIMULATIONS_EVAL = 1600
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
TEMPERATURE_THRESHOLD = 30   # moves before switching to low temperature
TEMP_HIGH = 1.0
TEMP_LOW = 0.1

# Self-play
NUM_WORKERS = 8              # parallel self-play processes
SELF_PLAY_GAMES = 100        # games per iteration
MAX_GAME_MOVES = 300         # draw after this many moves

# Replay buffer
REPLAY_BUFFER_SIZE = 500_000

# Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
LR_MILESTONES = [50, 75]     # iterations to decay LR
LR_GAMMA = 0.1
WEIGHT_DECAY = 1e-4
TRAINING_STEPS = 1000        # gradient steps per iteration

# Evaluation
EVAL_GAMES = 40              # arena games per evaluation
WIN_RATE_THRESHOLD = 0.55    # accept new model if it wins this fraction

# Checkpointing
NUM_ITERATIONS = 100
CHECKPOINT_DIR = "models"
LOG_DIR = "runs"
