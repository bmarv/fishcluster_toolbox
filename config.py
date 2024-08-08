from envbash import load_envbash
import os
import shutil

config_path = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(f"{config_path}/config.env"):
    shutil.copyfile(f"{config_path}/misc/config.env.default", "config.env")

load_envbash(f"{config_path}/config.env")
PROJ_PATH = os.environ["PROJ_PATH"]

N_BLOCKS=int(os.environ['N_BLOCKS'])
for block in range(N_BLOCKS):
    exec(f"BLOCK{block+1}=\"block{block+1}\"")
BLOCK = os.environ['BLOCK']
# THRESHOLDS for the data set filtered for erroneous frames
SPIKE_THRESHOLD = int(os.environ["SPIKE_THRESHOLD"])
DIRT_THRESHOLD = int(os.environ["DIRT_THRESHOLD"])
THRESHOLD_AREA_PX = int(os.environ["THRESHOLD_AREA_PX"])
# FILTERING
AREA_FILTER = int(os.environ["AREA_FILTER"])  # 1 to filter by area, 0 to not filter
DIRT_FILTER = int(os.environ["DIRT_FILTER"])  # 1 to filter by dirt, 0 to not filter
DIR_CSV_LOCAL =  os.environ["PROJ_PATH"] + f"/FE_tracks_060000_{BLOCK}"
FRONT, BACK = "front", "back"
CONFIG_DATA = os.environ['CONFIG_DATA']
config_path = f"{DIR_CSV_LOCAL}/" + os.environ["CONFIG_DATA"]

N_BATCHES = int(os.environ["N_BATCHES"])
MIN_BATCH_IDX=0 # SPECIFY THE MINIMUM BATCH-INDEX OF THE DAY TO INCLUDE
MAX_BATCH_IDX=N_BATCHES - 1 # SPECIFY THE MAXIMUM BATCH-INDEX OF THE DAY TO INCLUDE
BATCH_SIZE=10000  # Number of data frames per batch
FRAMES_PER_SECOND=5 # Number of frames per second

# Calibrations
# AREA CONFIG
area_back=f"{DIR_CSV_LOCAL}/area_config/areas_back"
area_front=f"{DIR_CSV_LOCAL}/area_config/areas_front"
CALIBRATION_DIST_CM = float(os.environ["CALIBRATION_DIST_CM"])
DEFAULT_CALIBRATION = float(os.environ["DEFAULT_CALIBRATION"])
err_file = f"{DIR_CSV_LOCAL}/results/log_error.csv"



