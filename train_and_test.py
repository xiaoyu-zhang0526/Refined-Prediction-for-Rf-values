import warnings
from modules.dataset import *
from modules.models import *
from modules.experiment import *
from modules.parameters import *
from modules.utils import *
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    config = parse_args()
    config.seed=324
    train_model(config)