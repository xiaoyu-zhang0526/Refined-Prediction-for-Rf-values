from modules.utils import *
from modules.parameters import *
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    smiles=[
        'O=C(OC1C(OC(C)=O)C(OC(C)=O)C(OC(C)=O)C(COC(C)=O)O1)C',
        'CC(OC[C@H]1O[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H](OC(C)=O)[C@H]1OC(C)=O)=O',
        'OC1O[C@H](COCC2=CC=CC=C2)[C@@H](OCC3=CC=CC=C3)[C@H](OCC4=CC=CC=C4)[C@H]1OCC5=CC=CC=C5',
        'CC(OC[C@@H]1[C@@H](OC(C)=O)[C@H](OC(C)=O)C=CO1)=O'
    ]
    diploe=[4.707,6.66,2.756,3.295]
    config = parse_args_predict()
    for i in range(4):
        predict_single(config,smiles[i],diploe[i])