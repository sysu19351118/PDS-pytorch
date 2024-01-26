import sys
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = '6'

sys.argv.extend(['--cfg_file', 'configs/sbd_snake.yaml', 'demo_path', "/data/tzx/data/images", 'ct_score', '0.3','ff_num','0'])
from run import run_demo
run_demo() 
