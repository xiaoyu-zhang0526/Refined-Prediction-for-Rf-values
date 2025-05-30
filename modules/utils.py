from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
import warnings
from modules.dataset import *
from modules.models import *
from modules.experiment import *
from modules.parameters import *
import pickle
warnings.filterwarnings("ignore")



def predict_single(config,smile,dipole=-1):
    # config = parse_args()
    Data = Dataset_process(config)
    Model = Model_ML(config)
    if dipole==-1:
        config.add_dipole = False
        model = load_model(config)
        compound_mol = Chem.MolFromSmiles(smile)
        Finger = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile))
        fingerprint = np.array([x for x in Finger])
        compound_finger = fingerprint
        compound_MolWt = Descriptors.ExactMolWt(compound_mol)
        compound_TPSA = Chem.rdMolDescriptors.CalcTPSA(compound_mol)
        # print(compound_TPSA)
        compound_nRotB = Descriptors.NumRotatableBonds(compound_mol)  # Number of rotable bonds
        compound_HBD = Descriptors.NumHDonors(compound_mol)  # Number of H bond donors
        compound_HBA = Descriptors.NumHAcceptors(compound_mol)  # Number of H bond acceptors
        compound_LogP = Descriptors.MolLogP(compound_mol)  # LogP
        X_test = np.zeros([1, 179])
        X_test[0, 0:167] = compound_finger
        X_test[0, 167:173] = 0
        X_test[0, 173:179] = [compound_MolWt, compound_TPSA, compound_nRotB, compound_HBD, compound_HBA, compound_LogP]
    else:
        config.add_dipole = True
        model = load_model(config)
        compound_mol = Chem.MolFromSmiles(smile)
        Finger = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile))
        fingerprint = np.array([x for x in Finger])
        compound_finger = fingerprint
        compound_MolWt = Descriptors.ExactMolWt(compound_mol)
        compound_TPSA = Chem.rdMolDescriptors.CalcTPSA(compound_mol)
        # print(compound_TPSA)
        compound_nRotB = Descriptors.NumRotatableBonds(compound_mol)  # Number of rotable bonds
        compound_HBD = Descriptors.NumHDonors(compound_mol)  # Number of H bond donors
        compound_HBA = Descriptors.NumHAcceptors(compound_mol)  # Number of H bond acceptors
        compound_LogP = Descriptors.MolLogP(compound_mol)  # LogP
        X_test = np.zeros([1, 180])
        X_test[0, 0:167] = compound_finger
        X_test[0, 167:173] = 0
        X_test[0,174]=dipole
        X_test[0, 174:180] = [compound_MolWt, compound_TPSA, compound_nRotB, compound_HBD, compound_HBA, compound_LogP]
    y=np.array([0.23]).reshape([1,1])
    X_test = X_test.copy()


    eluent_origin = np.array([config.eluent_origin], dtype=np.float32)
    eluent = []
    print(config.eluent_origin)
    for i in range(eluent_origin.shape[0]):
        eluent.append(Data.get_eluent_descriptor(eluent_origin[0]))
    eluent = np.array(eluent)

    X_test[0,167:173] = eluent[0]
    y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test.reshape(1, X_test.shape[1]),y[0],model)
    print(f'smile表达式: {smile}')
    print(f'使用的洗脱剂配比: {config.eluent_origin}')
    print(f'预测Rf值: {y_pred[0]}')



def train_model(config):
    save_path = os.path.join(config.model_save_path,config.use_model + '.pkl')
    Data = Dataset_process(config)
    X_train, y_train, X_validate, y_validate, X_test, y_test, data_array = Data.split_dataset()
    Model = Model_ML(config)
    model = Model.train(X_train, y_train, X_validate, y_validate)
    y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test, y_test, model)
    # save model with pickle
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to", save_path)
    print(R_square)

def load_model(config):
    model_path = config.model_save_path
    if not os.path.exists(model_path):
        config.model_save_path = os.path.dirname(config.model_save_path)
        train_model(config)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


    