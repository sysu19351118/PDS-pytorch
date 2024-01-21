import sys
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = '6'

sys.argv.extend(['--cfg_file', 'configs/sbd_snake.yaml', 'demo_path', "/data/tzx/snake_envo_num/visual_result/MRAVBCE/PDS-C/input", 'ct_score', '1.2','ff_num','0'])
#/home/amax/Titan_Five/TZX/snake-master/demo_images/2009_000871.jpg
#/home/amax/Titan_Five/TZX/snake-master/demo_images/4599_image.jpg
from run import run_visiual
run_visiual() 