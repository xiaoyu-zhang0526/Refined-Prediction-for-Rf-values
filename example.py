from modules.utils import *
from modules.parameters import *
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    config = parse_args_predict()
    config.smile_str='CC(C)=O'
    config.dipole_num=-1
    # Eluent_smiles=['CCCCCC','CC(OCC)=O','C(Cl)Cl','CO','CCOCC']
    config.eluent_origin = [0,0,0.95,0.05,0]
    predict_single(config,config.smile_str,config.dipole_num)