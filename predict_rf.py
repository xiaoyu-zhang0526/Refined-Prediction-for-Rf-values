from modules.utils import *
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    config = parse_args_predict()
    config.seed=324
    predict_single(config,config.smile_str,config.dipole_num)
