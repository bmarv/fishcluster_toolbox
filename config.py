from envbash import load_envbash
import os
import shutil

config_path = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(f"{config_path}/config.env"):
    shutil.copyfile(f"{config_path}/scripts/config.env.default", "config.env")

load_envbash(f"{config_path}/config.env")
N_FISHES = 24
BATCH_SIZE = 10000 
BACK = "back"
sep = ";"
VIS_DIR = "vis"
DIR_TRACES= "traces"
CAM_POS = "cam_pos"
DAY="day"
BATCH="batch"
DATAFRAME="DATAFRAME"
HOURS_PER_DAY = 8
FRAMES_PER_SECOND = 5
projectPath = os.environ["projectPath"]
BLOCK1 = os.environ["BLOCK1"]
BLOCK2 = os.environ["BLOCK2"]
BLOCK = os.environ["BLOCK"]
