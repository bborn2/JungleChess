# AlphaZero hyperparameters for JungleChess

# Board
BOARD_ROWS = 9
BOARD_COLS = 7
INPUT_CHANNELS = 17       # 8 blue planes + 8 red planes + 1 player plane
ACTION_SPACE = 8686       # from_row*1000 + from_col*100 + to_row*10 + to_col

# Network
NUM_RES_BLOCKS = 2   # smoke-test: was 10
NUM_FILTERS = 32     # smoke-test: was 128
POLICY_FILTERS = 16  # smoke-test: was 32
VALUE_FILTERS = 4

# MCTS
NUM_SIMULATIONS_TRAIN = 50   # smoke-test: was 800
NUM_SIMULATIONS_EVAL = 100   # smoke-test: was 1600
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
TEMPERATURE_THRESHOLD = 30   # moves before switching to low temperature
TEMP_HIGH = 1.0
TEMP_LOW = 0.1

# Self-play
NUM_WORKERS = 1              # smoke-test: was 8
SELF_PLAY_GAMES = 4          # smoke-test: was 100
MAX_GAME_MOVES = 100         # smoke-test: was 300

# Replay buffer
REPLAY_BUFFER_SIZE = 500_000

# Training
BATCH_SIZE = 32              # smoke-test: was 256
LEARNING_RATE = 1e-3
LR_MILESTONES = [50, 75]     # iterations to decay LR
LR_GAMMA = 0.1
WEIGHT_DECAY = 1e-4
TRAINING_STEPS = 10          # smoke-test: was 1000

# Evaluation
EVAL_GAMES = 4               # smoke-test: was 40
WIN_RATE_THRESHOLD = 0.55    # accept new model if it wins this fraction

# Checkpointing
NUM_ITERATIONS = 2           # smoke-test: was 100
CHECKPOINT_DIR = "models"
LOG_DIR = "runs"
