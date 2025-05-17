import argparse
import os
import torch

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default=os.path.join(os.getcwd(),'data','TLC_dataset.xlsx'), help='path of download dataset')
    parser.add_argument('--dipole_path', type=str, default=os.path.join(os.getcwd(),'data','dipole_moment_info.xlsx'),
                        help='path of dipole file')
    parser.add_argument('--data_range', type=int, default=4944, help='utilized data range,robot:4114,manual:4458,new:4944')
    parser.add_argument('--automatic_divide', type=bool, default=False, help='automatically divide dataset by 80% train,10% validate and 10% test')
    parser.add_argument('--choose_total', type=int, default=387, help='train total num,robot:387,manual:530')
    parser.add_argument('--choose_train', type=int, default=308, help='train num,robot:387,manual:530')
    parser.add_argument('--choose_validate', type=int, default=38, help='validate num')
    parser.add_argument('--choose_test', type=int, default=38, help='test num')
    parser.add_argument('--seed', type=int, default=324, help='random seed for split dataset')
    parser.add_argument('--torch_seed', type=int, default=324, help='random seed for torch')
    parser.add_argument('--add_dipole', type=bool, default=True, help='add dipole into dataset')
    parser.add_argument('--add_molecular_descriptors', type=bool, default=True, help='add molecular_descriptors (MW、TPSA、NROTB、HBA、HBD、LogP into dataset')
    parser.add_argument('--add_MACCkeys', type=bool, default=True,help='add MACCSkeys into dataset')
    parser.add_argument('--add_eluent_matrix', type=bool, default=True,help='add eluent matrix into dataset')
    parser.add_argument('--test_mode', type=str, default='robot', help='manual data or robot data or fix, costum test data')
    parser.add_argument('--use_model', type=str, default='Ensemble',help='the utilized model (XGB,LGB,ANN,RF,Ensemble,Bayesian)')
    parser.add_argument('--download_data', type=bool, default=False, help='use local dataset or download from dataset')
    parser.add_argument('--use_sigmoid', type=bool, default=True, help='use sigmoid')
    parser.add_argument('--shuffle_array', type=bool, default=True, help='shuffle_array')
    parser.add_argument('--characterization_mode', type=str, default='standard',
                        help='the characterization mode for the dataset, including standard, precise_TPSA, no_multi')

    #---------------parapmeters for plot---------------------
    parser.add_argument('--plot_col_num', type=int, default=4, help='The col_num in plot')
    parser.add_argument('--plot_row_num', type=int, default=4, help='The row_num in plot')
    parser.add_argument('--plot_importance_num', type=int, default=10, help='The max importance num in plot')
    #--------------parameters For LGB-------------------
    parser.add_argument('--LGB_max_depth', type=int, default=5, help='max_depth for LGB')
    parser.add_argument('--LGB_num_leaves', type=int, default=25, help='num_leaves for LGB')
    parser.add_argument('--LGB_learning_rate', type=float, default=0.007, help='learning_rate for LGB')
    parser.add_argument('--LGB_n_estimators', type=int, default=1000, help='n_estimators for LGB')
    parser.add_argument('--LGB_early_stopping_rounds', type=int, default=200, help='early_stopping_rounds for LGB')

    #---------------parameters for XGB-----------------------
    parser.add_argument('--XGB_n_estimators', type=int, default=200, help='n_estimators for XGB')
    parser.add_argument('--XGB_max_depth', type=int, default=3, help='max_depth for XGB')
    parser.add_argument('--XGB_learning_rate', type=float, default=0.1, help='learning_rate for XGB')

    #---------------parameters for RF------------------------
    parser.add_argument('--RF_n_estimators', type=int, default=1000, help='n_estimators for RF')
    parser.add_argument('--RF_random_state', type=int, default=1, help='random_state for RF')
    parser.add_argument('--RF_n_jobs', type=int, default=1, help='n_jobs for RF')

    #--------------parameters for ANN-----------------------
    parser.add_argument('--NN_hidden_neuron', type=int, default=128, help='hidden neurons for NN')
    parser.add_argument('--NN_optimizer', type=str, default='Adam', help='optimizer for NN (Adam,SGD,RMSprop)')
    parser.add_argument('--NN_lr', type=float, default=0.005, help='learning rate for NN')
    parser.add_argument('--NN_model_save_location', type=str, default=os.path.join(os.getcwd(),'model_save_NN'), help='learning rate for NN')
    parser.add_argument('--NN_max_epoch', type=int, default=5000, help='max training epoch for NN')
    parser.add_argument('--NN_add_sigmoid', type=bool, default=True, help='whether add sigmoid in NN')
    parser.add_argument('--NN_add_PINN', type=bool, default=False, help='whether add PINN in NN')
    parser.add_argument('--NN_epi', type=float, default=100.0, help='The coef of PINN Loss in NN')
    return parser

def parse_args():
    parser = init_args()
    parser.add_argument('--model_save_path', type=str, default=os.path.join(os.getcwd(),'model'), help='path of model save')
    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config

def parse_args_predict():
    parser = init_args()
    parser.add_argument('--model_load_path', type=str, default=os.path.join(os.getcwd(),'model/model.pkl'), help='path of model load')
    parser.add_argument('--smile_str', type=str, default='O=C(OC1C(OC(C)=O)C(OC(C)=O)C(OC(C)=O)C(COC(C)=O)O1)C', help='smile string')
    parser.add_argument('--dipole_num', type=float, default=4.707, help='dipole moment')
    
    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config