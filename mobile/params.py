from datetime import datetime
from re import sub

TIMESTAMP = sub('[-: ]', '', str(datetime.today()).split('.')[0])
MODEL_SAVE_PATH = "models\\" + TIMESTAMP + '.h5'
IMG_DATA_PATH = "C:\\Users\\Yunxuan\\OneDrive - National University of Singapore"
CSV_TRAIN_PATH = "ndsc-advanced\\mobile_data_info_train_competition.csv"
CSV_VALID_PATH = "ndsc-advanced\\mobile_data_info_val_competition.csv"
MODEL_SAVE_PATH = "models\\" + TIMESTAMP + ".h5"
LABEL_TYPES = ['Operating System', 'Features', 'Network Connections', 'Memory RAM', 'Brand',
               'Warranty Period', 'Storage Capacity', 'Color Family', 'Phone Model', 'Camera', 'Phone Screen Size']
LABEL_DIMS = [7, 7, 4, 10, 56, 14, 18, 26, 2280, 15, 6]

VALIDATION_SPLIT = .1

IMG_SIZE = (128, 128)

EMBEDDING_DIM = 16
LR_SCHEDULER_DURATION = 30
LR_SCHEDULER_END = 40
LR_START = 1e-3
LR_END = 1e-5
BATCH_SIZE = 64
N_EPOCHS = 120