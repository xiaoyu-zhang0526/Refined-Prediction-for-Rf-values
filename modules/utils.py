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
    if dipole==-1:
        config.add_dipole = False
        Data = Dataset_process(config)
        X_train, y_train, X_validate, y_validate, X_test, y_test, data_array = Data.split_dataset()
        Model = Model_ML(config)
        model = Model.train(X_train, y_train, X_validate, y_validate)
        y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test, y_test, data_array, model)

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
        Data = Dataset_process(config)
        X_train, y_train, X_validate, y_validate, X_test, y_test, data_array = Data.split_dataset()
        Model = Model_ML(config)
        model = Model.train(X_train, y_train, X_validate, y_validate)
        y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test, y_test, data_array, model)
        # print(R_square)

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
    eluent_origin = np.array([[1, 0, 0, 0, 0], [0.980392, 0.019608, 0, 0, 0], [0.952381, 0.047619, 0, 0, 0],
                  [0.833333, 0.166667, 0, 0, 0], [0.75, 0.25, 0, 0, 0], [0.5, 0.5, 0, 0, 0],
                  [0.333333, 0.666667, 0, 0, 0], [0, 1, 0, 0, 0],[0, 0, 1, 0, 0], [0, 0, 0.990099, 0.009901, 0], [0, 0, 0.980392, 0.019608, 0],
                        [0, 0, 0.967742, 0.032258, 0], [0, 0, 0.952381, 0.047619, 0], [0, 0, 0.909091, 0.090909, 0]], dtype=np.float32)
    eluent = []

    for i in range(eluent_origin.shape[0]):
        eluent.append(Data.get_eluent_descriptor(eluent_origin[i]))
    eluent = np.array(eluent)
    print(smile)
    for i in range(eluent.shape[0]):
        X_test[0,167:173] = eluent[i]
        y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test.reshape(1, X_test.shape[1]),
                                                        y[0],
                                                        data_array, model)

        print(y_pred[0])


def train_model(config):
    config.add_dipole = True
    save_path = os.path.join(config.model_save_path,config.use_model + '.pkl')
    Data = Dataset_process(config)
    X_train, y_train, X_validate, y_validate, X_test, y_test, data_array = Data.split_dataset()
    Model = Model_ML(config)
    model = Model.train(X_train, y_train, X_validate, y_validate)
    y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test, y_test, data_array, model)
    # save model with pickle
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to", save_path)
    print(R_square)


    