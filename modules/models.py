from sklearn import linear_model
from matplotlib.ticker import MultipleLocator
import  os
import pandas as pd
from rdkit import Chem
import numpy as np
from rdkit.Chem import Draw
from xgboost.sklearn import XGBRegressor
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import time
import matplotlib.pyplot as plt
import warnings
from .dataset import *

warnings.filterwarnings("ignore")

font1 = {'family': 'Arial',
         'weight': 'normal',
         #"style": 'italic',
         'size': 14,
         }
font1egend = {'family': 'Arial',
         'weight': 'normal',
         #"style": 'italic',
         'size': 5,
         }

class ANN(nn.Module):
    '''
    Construct artificial neural network
    '''
    def __init__(self, in_neuron, hidden_neuron, out_neuron,config):
        super(ANN, self).__init__()
        self.input_layer = nn.Linear(in_neuron, hidden_neuron)
        self.hidden_layer = nn.Linear(hidden_neuron, hidden_neuron)
        self.output_layer = nn.Linear(hidden_neuron, out_neuron)
        self.NN_add_sigmoid=config.NN_add_sigmoid


    def forward(self, x):
        x = self.input_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.output_layer(x)
        if self.NN_add_sigmoid==True:
            x = F.sigmoid(x)
        return x

class Model_ML():
    def __init__(self,config):
        super(Model_ML, self).__init__()
        self.seed=config.seed
        self.torch_seed=config.seed
        self.config=config
        self.data_range=config.data_range
        self.file_path=config.file_path
        self.choose_train = config.choose_train
        self.choose_validate = config.choose_validate
        self.choose_test = config.choose_test
        self.add_dipole = config.add_dipole
        self.add_molecular_descriptors = config.add_molecular_descriptors
        self.add_eluent_matrix=config.add_eluent_matrix
        self.use_sigmoid=config.use_sigmoid

        self.use_model=config.use_model
        self.LGB_max_depth=config.LGB_max_depth
        self.LGB_num_leaves=config.LGB_num_leaves
        self.LGB_learning_rate=config.LGB_learning_rate
        self.LGB_n_estimators=config.LGB_n_estimators
        self.LGB_early_stopping_rounds=config.LGB_early_stopping_rounds

        self.XGB_n_estimators=config.XGB_n_estimators
        self.XGB_max_depth = config.XGB_max_depth
        self.XGB_learning_rate = config.XGB_learning_rate

        self.RF_n_estimators=config.RF_n_estimators
        self.RF_random_state=config.RF_random_state
        self.RF_n_jobs=config.RF_n_jobs

        self.NN_hidden_neuron=config.NN_hidden_neuron
        self.NN_optimizer=config.NN_optimizer
        self.NN_lr= config.NN_lr
        self.NN_model_save_location=config.NN_model_save_location
        self.NN_max_epoch=config.NN_max_epoch
        self.NN_add_PINN=config.NN_add_PINN
        self.NN_epi=config.NN_epi
        self.device=config.device

        self.plot_row_num=config.plot_row_num
        self.plot_col_num=config.plot_col_num
        self.plot_importance_num=config.plot_importance_num

    def train(self,X_train,y_train,X_validate,y_validate):


        '''
        train model using LightGBM,Xgboost,Random forest or ANN
        '''
        print('----------Start Training!--------------')
        torch.manual_seed(self.torch_seed)
        if self.use_model=='LGB':
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model = lgb.LGBMRegressor(objective='regression', max_depth=self.LGB_max_depth,
                                      num_leaves=self.LGB_num_leaves,
                                      learning_rate=self.LGB_learning_rate, n_estimators=self.LGB_n_estimators)
            model.fit(X_train, list(y_train.reshape(y_train.shape[0], )),
                      eval_set=[(X_train, list(y_train.reshape(y_train.shape[0], ))),
                                (X_validate, list(y_validate.reshape(y_validate.shape[0], )))],
                      eval_names=('fit', 'val'), eval_metric='l2', early_stopping_rounds=self.LGB_early_stopping_rounds,
                      verbose=False)
            print('----------LGB Training Finished!--------------')
            return model
        elif self.use_model=='XGB':
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model = XGBRegressor(seed=self.seed,
                                 n_estimators=self.XGB_n_estimators,
                                 max_depth=self.XGB_max_depth,
                                 eval_metric='rmse',
                                 learning_rate=self.XGB_learning_rate,
                                 min_child_weight=1,
                                 subsample=1,
                                 colsample_bytree=1,
                                 colsample_bylevel=1,
                                 gamma=0)

            model.fit(X_train, y_train.reshape(y_train.shape[0]))
            print('----------XGB Training Finished!--------------')
            return model

        elif self.use_model=='RF':
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model = RandomForestRegressor(n_estimators=self.RF_n_estimators,
                                          criterion='squared_error',
                                          random_state=self.RF_random_state,
                                          n_jobs=self.RF_n_jobs)
            model.fit(X_train, y_train)
            print('----------RF Training Finished!--------------')
            return model

        elif self.use_model=='ANN':
            Net = ANN(X_train.shape[1], self.NN_hidden_neuron, 1, config=self.config).to(self.device)
            X_train = Variable(torch.from_numpy(X_train.astype(np.float32)).to(self.device), requires_grad=True)
            y_train = Variable(torch.from_numpy(y_train.astype(np.float32)).to(self.device))
            X_validate = Variable(torch.from_numpy(X_validate.astype(np.float32)).to(self.device), requires_grad=True)
            y_validate = Variable(torch.from_numpy(y_validate.astype(np.float32)).to(self.device))

            model_name = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
            dir_name = self.NN_model_save_location + '/' + model_name
            loss_plot = []
            loss_validate_plot = []
            try:
                os.makedirs(dir_name)
            except OSError:
                pass

            if self.NN_optimizer == 'SGD':
                optimizer = torch.optim.SGD(Net.parameters(), lr=self.NN_lr)
            elif self.NN_optimizer == 'RMSprop':
                optimizer = torch.optim.RMSprop(Net.parameters(), lr=self.NN_lr)
            else:
                optimizer = torch.optim.Adam(Net.parameters(), lr=self.NN_lr)

            with open(dir_name + '/' + 'data.txt', 'w') as f:  # 设置文件对象
                for epoch in range(self.NN_max_epoch):
                    optimizer.zero_grad()
                    prediction = Net(X_train)
                    prediction_validate = Net(X_validate)
                    dprediction = \
                    torch.autograd.grad(outputs=prediction[:, 0].sum(), inputs=X_train, create_graph=True)[0]
                    dprediction_deluent = dprediction[:, 167].reshape(X_train.shape[0], 1)
                    dprediction_validate = \
                    torch.autograd.grad(outputs=prediction_validate[:, 0].sum(), inputs=X_validate,
                                        create_graph=True)[0]
                    dprediction_validate_deluent = dprediction_validate[:, 167].reshape(X_validate.shape[0], 1)
                    MSELoss=torch.nn.MSELoss()
                    if self.NN_add_PINN == True:
                        loss = MSELoss(y_train, prediction) + self.NN_epi * (
                            torch.sum(F.relu(-dprediction_deluent))) / X_train.shape[0]
                        loss_validate = MSELoss(y_validate, prediction_validate) + (self.NN_epi * torch.sum(
                            F.relu(-dprediction_validate_deluent))) / X_validate.shape[0]
                    else:
                        loss = MSELoss(y_train, prediction)
                        loss_validate = MSELoss(y_validate, prediction_validate)

                    loss.backward()
                    optimizer.step()
                    if epoch>200:
                        if loss.item() == loss_plot[-1]:
                            if self.NN_optimizer == 'SGD':
                                optimizer = torch.optim.SGD(Net.parameters(), lr=self.NN_lr * 0.2)
                            elif self.NN_optimizer == 'RMSprop':
                                optimizer = torch.optim.RMSprop(Net.parameters(), lr=self.NN_lr * 0.2)
                            else:
                                optimizer = torch.optim.Adam(Net.parameters(), lr=self.NN_lr * 0.2)

                    if (epoch + 1) % 100 == 0:
                        print("iter_num: %d      loss: %.8f    loss_validate: %.8f" % (
                        epoch + 1, loss.item(), loss_validate.item()))
                        f.write("iter_num: %d      loss: %.8f    loss_validate: %.8f \r\n" % (
                        epoch + 1, loss.item(), loss_validate.item()))
                        torch.save(Net.state_dict(), dir_name + '/' + "%d_epoch.pkl" % (epoch+1))
                        loss_plot.append(loss.item())
                        loss_validate_plot.append(loss_validate.item())

                best_epoch=(loss_validate_plot.index(min(loss_validate_plot))+1)*100
                print("The ANN has been trained, the best epoch is %d"%(best_epoch))
                Net.load_state_dict(torch.load(dir_name + '/' + "%d_epoch.pkl" % (best_epoch)))
                Net.eval()

                plt.figure(3)
                plt.plot(loss_plot, marker='x', label='loss')
                plt.plot(loss_validate_plot, c='red', marker='v', label='loss_validate')
                plt.legend()
                plt.savefig(dir_name + '/' + 'loss_pic.png')
                print('----------ANN Training Finished!--------------')
            return Net

        elif self.use_model=='Bayesian':
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            clf = linear_model.BayesianRidge()
            clf.fit(X_train, y_train.reshape(y_train.shape[0]))
            return clf

        elif self.use_model=='Ensemble':
            y_train_origin=y_train.copy()
            y_validate_origin = y_validate.copy()
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model_LGB = lgb.LGBMRegressor(objective='regression', max_depth=self.LGB_max_depth,
                                      num_leaves=self.LGB_num_leaves,
                                      learning_rate=self.LGB_learning_rate, n_estimators=self.LGB_n_estimators)
            model_LGB.fit(X_train, list(y_train.reshape(y_train.shape[0], )),
                      eval_set=[(X_train, list(y_train.reshape(y_train.shape[0], ))),
                                (X_validate, list(y_validate.reshape(y_validate.shape[0], )))],
                      eval_names=('fit', 'val'), eval_metric='l2', early_stopping_rounds=self.LGB_early_stopping_rounds,
                      verbose=False)
            model_XGB = XGBRegressor(seed=self.seed,
                                 n_estimators=self.XGB_n_estimators,
                                 max_depth=self.XGB_max_depth,
                                 eval_metric='rmse',
                                 learning_rate=self.XGB_learning_rate,
                                 min_child_weight=1,
                                 subsample=1,
                                 colsample_bytree=1,
                                 colsample_bylevel=1,
                                 gamma=0)

            model_XGB.fit(X_train, y_train.reshape(y_train.shape[0]))
            model_RF = RandomForestRegressor(n_estimators=self.RF_n_estimators,
                                          criterion='squared_error',
                                          random_state=self.RF_random_state,
                                          n_jobs=self.RF_n_jobs)
            model_RF.fit(X_train, y_train)

            Net = ANN(X_train.shape[1], self.NN_hidden_neuron, 1, config=self.config).to(self.device)
            X_train = Variable(torch.from_numpy(X_train.astype(np.float32)).to(self.device), requires_grad=True)
            y_train = Variable(torch.from_numpy(y_train_origin.astype(np.float32)).to(self.device))
            X_validate = Variable(torch.from_numpy(X_validate.astype(np.float32)).to(self.device), requires_grad=True)
            y_validate = Variable(torch.from_numpy(y_validate_origin.astype(np.float32)).to(self.device))

            model_name = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
            dir_name = self.NN_model_save_location + '/' + model_name
            loss_plot = []
            loss_validate_plot = []
            try:
                os.makedirs(dir_name)
            except OSError:
                pass

            if self.NN_optimizer == 'SGD':
                optimizer = torch.optim.SGD(Net.parameters(), lr=self.NN_lr)
            elif self.NN_optimizer == 'RMSprop':
                optimizer = torch.optim.RMSprop(Net.parameters(), lr=self.NN_lr)
            else:
                optimizer = torch.optim.Adam(Net.parameters(), lr=self.NN_lr)

            with open(dir_name + '/' + 'data.txt', 'w') as f:  # 设置文件对象
                for epoch in range(self.NN_max_epoch):
                    optimizer.zero_grad()
                    prediction = Net(X_train)
                    prediction_validate = Net(X_validate)
                    MSELoss = torch.nn.MSELoss()
                    if self.NN_add_PINN == True:
                        dprediction = \
                            torch.autograd.grad(outputs=prediction[:, 0].sum(), inputs=X_train, create_graph=True)[0]
                        dprediction_deluent = dprediction[:, 167].reshape(X_train.shape[0], 1)
                        dprediction_validate = \
                            torch.autograd.grad(outputs=prediction_validate[:, 0].sum(), inputs=X_validate,
                                                create_graph=True)[0]
                        dprediction_validate_deluent = dprediction_validate[:, 167].reshape(X_validate.shape[0], 1)
                        loss = MSELoss(y_train, prediction) + self.NN_epi * (
                            torch.sum(F.relu(-dprediction_deluent))) / X_train.shape[0]
                        loss_validate = MSELoss(y_validate, prediction_validate) + (self.NN_epi * torch.sum(
                            F.relu(-dprediction_validate_deluent))) / X_validate.shape[0]
                    else:
                        loss = MSELoss(y_train, prediction)
                        loss_validate = MSELoss(y_validate, prediction_validate)

                    loss.backward()
                    optimizer.step()


                    if (epoch + 1) % 100 == 0:
                        print("iter_num: %d      loss: %.8f    loss_validate: %.8f" % (
                            epoch + 1, loss.item(), loss_validate.item()))
                        f.write("iter_num: %d      loss: %.8f    loss_validate: %.8f \r\n" % (
                            epoch + 1, loss.item(), loss_validate.item()))
                        torch.save(Net.state_dict(), dir_name + '/' + "%d_epoch.pkl" % (epoch + 1))
                        loss_plot.append(loss.item())
                        loss_validate_plot.append(loss_validate.item())

                best_epoch = (loss_validate_plot.index(min(loss_validate_plot)) + 1) * 100
                print("The ANN has been trained, the best epoch is %d" % (best_epoch))
                Net.load_state_dict(torch.load(dir_name + '/' + "%d_epoch.pkl" % (best_epoch)))
                Net.eval()

                plt.figure(100)
                plt.plot(loss_plot, marker='x', label='loss')
                plt.plot(loss_validate_plot, c='red', marker='v', label='loss_validate')
                plt.legend()
                plt.savefig(dir_name + '/' + 'loss_pic.png')
            return model_LGB,model_XGB,model_RF,Net

    def plot_total_variance(self,y_test,y_pred):
        '''
        plot and calculate the MSE, RMSE, MAE and R^2
        '''
        if self.use_model=='ANN':
            y_test=y_test.cpu().data.numpy()

        y_test=y_test.reshape(y_test.shape[0])
        y_pred=y_pred.reshape(y_pred.shape[0])

        # y_plot=np.hstack((y_test.reshape(y_test.shape[0],1),y_pred.reshape(y_pred.shape[0],1)))
        # df=pd.DataFrame(y_plot)
        # df.columns = ['True_value', 'Prediction_value']
        # df['method']=self.use_model
        #print(df)
        #df.to_csv(f"result_save/revised_{self.use_model}_compound.csv")
        #
        ## ------------plot total loss---------------
        # print(df)
        # base_plot = (ggplot(df) +geom_point(aes('True_value', 'Prediction_value'),alpha=0.3,color="blue",size=4)
        #               +geom_line(aes('True_value','True_value'),linetype='--',size=1)+ggtitle(self.use_model)+
        #               xlab('Observed Yield')+ylab('Prediction Yield'))#+theme(axis_text=element_text(size=16)))
        # print(base_plot)

        MSE = np.sum(np.abs(y_test - y_pred)**2) /y_test.shape[0]
        RMSE=np.sqrt(MSE)
        MAE = np.sum(np.abs(y_test - y_pred)) / y_test.shape[0]
        R_square=1-(((y_test-y_pred)**2).sum()/((y_test-y_test.mean())**2).sum())
        #print(f"MSE is {MSE}, RMSE is {RMSE}, MAE is {MAE}, R_square is {R_square}")
        return MSE,RMSE,MAE,R_square

    def plot_predict_polarity(self,X_test,y_test,data_array,model):
        '''
        plot prediction for each compound in the test dataset
        '''
        plt.style.use('ggplot')
        data_range=self.data_range
        if self.use_model=='ANN':
            X_test=X_test.cpu().data.numpy()
            y_test = y_test.cpu().data.numpy()

        Dataset_process.download_dataset(self,print_info=False)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info=entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]
        compound_Rf = compound_info[:, 9]
        compound_eluent = np.array(compound_info[:, 4:9], dtype=np.float32)


        N = 8
        x_origin = np.array([[0,1,0,0,0],[0.333333,0.666667,0,0,0],[0.5,0.5,0,0,0],
                      [0.75,0.25,0,0,0],[0.833333,0.166667,0,0,0],[0.952381,0.047619,0,0,0],
                      [0.980392,0.019608,0,0,0],[1,	0,	0,	0,0]],dtype=np.float32)
        x_ME_origin=np.array([[0,0,1,0,0],[0,0,0.990099,0.009901,0],[0,0,0.980392,0.019608,0],
                      [0,0,0.967742,0.032258,0],[0,0,0.952381,0.047619,0],[0,0,0.909091,0.090909,0]],dtype=np.float32)
        x_Et_origin=np.array([[0.66667,0,0,0,0.33333],[0.5,0,0,0,0.5],[0,0,0,0,1]])
        x=[]
        x_ME=[]
        x_Et=[]
        for i in range(x_origin.shape[0]):
            x.append(Dataset_process.get_eluent_descriptor(self,x_origin[i]))
        x=np.array(x)

        for i in range(x_ME_origin.shape[0]):
            x_ME.append(Dataset_process.get_eluent_descriptor(self,x_ME_origin[i]))
        x_ME=np.array(x_ME)

        for i in range(x_Et_origin.shape[0]):
            x_Et.append(Dataset_process.get_eluent_descriptor(self,x_Et_origin[i]))
        x_Et=np.array(x_Et)

        X_test_origin=X_test.copy()
        X_test[:,167:173]=0.0
        unique_rows,inv = np.unique(X_test.astype("<U22"),axis=0,return_inverse=True)

        index = data_array[
               self.choose_train + self.choose_validate:self.choose_train + self.choose_validate + self.choose_test]

        # if self.plot_col_num*self.plot_row_num!=unique_rows.shape[0]:
        #     raise Warning("col_num*row_num should equal to choose_test")

        for j in range(unique_rows.shape[0]):
            database_test = np.zeros([N, unique_rows.shape[1]])
            database_test_ME = np.zeros([6, unique_rows.shape[1]])
            database_test_Et = np.zeros([3, unique_rows.shape[1]])
            index_inv=np.where(inv==j)[0]
            a=len(np.unique(np.array(inv[0:index_inv[0]+1]).astype("<U22"),axis=0))-1
            print(index[a])
            ID_loc = np.where(compound_ID == index[a])[0]
            smiles = compound_smile[ID_loc[0]]
            eluents=compound_eluent[ID_loc]
            Rfs=compound_Rf[ID_loc]


            mol = Chem.MolFromSmiles(smiles)

            for i in range(N):
                database_test[i]=X_test_origin[index_inv[0]]
                database_test[i,167:173]=x[i]
            for i in range(6):
                database_test_ME[i]=X_test_origin[index_inv[0]]
                database_test_ME[i,167:173]=x_ME[i]
            for i in range(3):
                database_test_Et[i]=X_test_origin[index_inv[0]]
                database_test_Et[i,167:173]=x_ME[i]
            if self.use_model=='ANN':
                database_test=Variable(torch.from_numpy(database_test.astype(np.float32)).to(self.device), requires_grad=True)
                y_pred=model(database_test).cpu().data.numpy()
                y_pred =y_pred.reshape(y_pred.shape[0],)

                database_test_ME=Variable(torch.from_numpy(database_test_ME.astype(np.float32)).to(self.device), requires_grad=True)
                y_pred_ME=model(database_test_ME).cpu().data.numpy()
                y_pred_ME =y_pred_ME.reshape(y_pred_ME.shape[0],)

                database_test_Et = Variable(torch.from_numpy(database_test_Et.astype(np.float32)).to(self.device),
                                            requires_grad=True)
                y_pred_Et = model(database_test_Et).cpu().data.numpy()
                y_pred_Et = y_pred_Et.reshape(y_pred_Et.shape[0], )

            elif self.use_model=='Ensemble':
                model_LGB,model_XGB,model_RF,Net=model

                y_pred_LGB = model_LGB.predict(database_test)
                y_pred_ME_LGB = model_LGB.predict(database_test_ME)
                y_pred_Et_LGB = model_LGB.predict(database_test_Et)
                y_pred_XGB = model_XGB.predict(database_test)
                y_pred_ME_XGB = model_XGB.predict(database_test_ME)
                y_pred_Et_XGB = model_XGB.predict(database_test_Et)
                y_pred_RF = model_RF.predict(database_test)
                y_pred_ME_RF = model_RF.predict(database_test_ME)
                y_pred_Et_RF=model_RF.predict(database_test_Et)
                if self.use_sigmoid == True:
                    y_pred_LGB= 1 / (1 + np.exp(-y_pred_LGB))
                    y_pred_ME_LGB = 1 / (1 + np.exp(-y_pred_ME_LGB))
                    y_pred_Et_LGB = 1 / (1 + np.exp(-y_pred_Et_LGB))
                    y_pred_XGB= 1 / (1 + np.exp(-y_pred_XGB))
                    y_pred_ME_XGB = 1 / (1 + np.exp(-y_pred_ME_XGB))
                    y_pred_Et_XGB = 1 / (1 + np.exp(-y_pred_Et_XGB))
                    y_pred_RF= 1 / (1 + np.exp(-y_pred_RF))
                    y_pred_ME_RF = 1 / (1 + np.exp(-y_pred_ME_RF))
                    y_pred_Et_RF = 1 / (1 + np.exp(-y_pred_Et_RF))


                database_test = Variable(torch.from_numpy(database_test.astype(np.float32)).to(self.device),
                                         requires_grad=True)
                y_pred = Net(database_test).cpu().data.numpy()
                y_pred_NN = y_pred.reshape(y_pred.shape[0], )

                database_test_ME = Variable(torch.from_numpy(database_test_ME.astype(np.float32)).to(self.device),
                                            requires_grad=True)
                y_pred_ME = Net(database_test_ME).cpu().data.numpy()
                y_pred_ME_NN = y_pred_ME.reshape(y_pred_ME.shape[0], )

                database_test_Et = Variable(torch.from_numpy(database_test_Et.astype(np.float32)).to(self.device),
                                            requires_grad=True)
                y_pred_Et = Net(database_test_Et).cpu().data.numpy()
                y_pred_Et_NN = y_pred_Et.reshape(y_pred_Et.shape[0], )

                y_pred_ME=(0.2*y_pred_ME_LGB+0.2*y_pred_ME_XGB+0.2*y_pred_ME_RF+0.4*y_pred_ME_NN)
                y_pred_Et=(0.2*y_pred_Et_LGB+0.2*y_pred_Et_XGB+0.2*y_pred_Et_RF+0.4*y_pred_Et_NN)
                y_pred=(0.2*y_pred_NN+0.2*y_pred_XGB+0.2*y_pred_RF+0.4*y_pred_LGB)


            else:
                y_pred=model.predict(database_test)
                y_pred_ME = model.predict(database_test_ME)
                y_pred_Et = model.predict(database_test_Et)
                if self.use_sigmoid==True:
                    y_pred=1/(1+np.exp(-y_pred))
                    y_pred_ME = 1 / (1 + np.exp(-y_pred_ME))
                    y_pred_Et = 1 / (1 + np.exp(-y_pred_Et))



            EA_plot=[]
            y_EA_plot=[]
            Me_plot=[]
            y_ME_plot=[]
            Et_plot=[]
            y_Et_plot=[]
            for k in range(eluents.shape[0]):
                if eluents[k,1]+eluents[k,0]+eluents[k,4]==0:
                    Me_plot.append(np.log(np.array(eluents[k,3] + 1, dtype=np.float32)))
                    y_ME_plot.append(Rfs[k])
                if eluents[k, 2] + eluents[k, 3] + eluents[k, 4] == 0:
                    EA_plot.append(np.log(np.array(eluents[k, 1] + 1, dtype=np.float32)))
                    y_EA_plot.append(Rfs[k])
                if eluents[k, 1] + eluents[k,2] + eluents[k, 3] == 0 and eluents[k,4]!=0:
                    Et_plot.append(np.log(np.array(eluents[k, 4] + 1, dtype=np.float32)))
                    y_Et_plot.append(Rfs[k])

            #plt.style.use('ggplot')
            plt.figure(1,figsize=(2,2),dpi=300)
            ax=plt.subplot(1,1,1)
            #plt.scatter(Me_plot,y_ME_plot,c='pink',marker='v',s=200)
            plt.scatter(EA_plot,y_EA_plot,c='red',marker='^',label='True $R_f$ values',s=30,zorder=1)
            plt.plot(np.log(x_origin[:,1] + 1), y_pred,marker='x',markersize=5,linewidth=1, label='predicted $R_f$ curve',color='blue',zorder=2)
            #plt.plot(np.log(x_ME[:,3] + 1), y_pred_ME,linewidth=3, label='predict Rf curve',color='pink')
            # plt.scatter(np.log(np.array(X_test_origin[index_inv][:,167] + 1, dtype=np.float32)), y_test[index_inv], c='red',
            #             marker='^',s=200, label='True Rf')
            #plt.plot(np.log(x[:,0] + 1), y_pred,marker='o',markersize=10,linewidth=3, label='predict Rf curve')

            # plt.xlabel('Log EA ratio',font1)
            # plt.ylabel('Rf',font1)
            xmajorLocator = MultipleLocator(0.2)
            ax.xaxis.set_major_locator(xmajorLocator)
            plt.yticks(fontproperties='Arial', size=7)
            plt.xticks(fontproperties='Arial', size=7)
            plt.legend(loc='lower right', prop=font1egend)
            plt.ylim(-0.1, 1.1)

            plt.savefig(f'PPT_fig/PE_EA_{int(index[a])}.tiff',
                        bbox_inches='tight', dpi=300)
            plt.savefig(f'PPT_fig/PE_EA_{int(index[a])}.pdf',
                        bbox_inches='tight', dpi=300)
            plt.cla()
            # plt.tight_layout()

            plt.figure(2, figsize=(2, 2), dpi=300)
            smiles_pic = Draw.MolToImage(mol, size=(500, 500), kekulize=True)
            plt.axis('off')
            plt.imshow(smiles_pic)
            plt.savefig(f'PPT_fig/PE_EA_mol_{int(index[a])}.png',
                       bbox_inches='tight', dpi=300)
            plt.savefig(f'PPT_fig/PE_EA_mol_{int(index[a])}.pdf',
                        bbox_inches='tight', dpi=300)
            plt.cla()
            # plt.legend()

            plt.figure(3,figsize=(2,2),dpi=300)
            ax=plt.subplot(1,1,1)
            plt.scatter(Me_plot, y_ME_plot, c='green', marker='^',label='True $R_f$ values', s=30)
            #plt.scatter(EA_plot, y_EA_plot, c='red', marker='^', s=200)
            #plt.plot(np.log(x[:, 1] + 1), y_pred, linewidth=3, label='predict Rf curve', color='red')
            plt.plot(np.log(x_ME[:, 3] + 1), y_pred_ME, marker='x',markersize=5,linewidth=1, label='predicted $R_f$ curve', color='blue')
            # plt.scatter(np.log(np.array(X_test_origin[index_inv][:,167] + 1, dtype=np.float32)), y_test[index_inv], c='red',
            #             marker='^',s=200, label='True Rf')
            # plt.plot(np.log(x[:,0] + 1), y_pred,marker='o',markersize=10,linewidth=3, label='predict Rf curve')

            # plt.xlabel('Log MeOH ratio',font1)
            # plt.ylabel('Rf',font1)
            plt.legend(loc='lower right', prop=font1egend)
            xmajorLocator = MultipleLocator(0.02)
            ax.xaxis.set_major_locator(xmajorLocator)
            plt.yticks(fontproperties='Arial', size=7)
            plt.xticks(fontproperties='Arial', size=7)
            plt.ylim(-0.1, 1.1)
            plt.savefig("temp_fig.png", dpi=300)
            plt.savefig(f'PPT_fig/DCM_MeOH_{int(index[a])}.tiff',
                        bbox_inches='tight', dpi=300)
            plt.savefig(f'PPT_fig/DCM_MeOH_{int(index[a])}.pdf',
                        bbox_inches='tight', dpi=300)
            plt.cla()



            plt.figure(4,figsize=(2,2),dpi=300)
            ax=plt.subplot(1,1,1)
            plt.scatter(Et_plot, y_Et_plot, c='orange',label='True $R_f$ values', marker='^', s=30)
            # plt.scatter(EA_plot, y_EA_plot, c='red', marker='^', s=200)
            # plt.plot(np.log(x[:, 1] + 1), y_pred, linewidth=3, label='predict Rf curve', color='red')
            plt.plot(np.log(x_Et[:, 4] + 1), y_pred_Et, marker='x', markersize=5, linewidth=1,
                     label='predict $R_f$ curve', color='blue')
            # plt.scatter(np.log(np.array(X_test_origin[index_inv][:,167] + 1, dtype=np.float32)), y_test[index_inv], c='red',
            #             marker='^',s=200, label='True Rf')
            # plt.plot(np.log(x[:,0] + 1), y_pred,marker='o',markersize=10,linewidth=3, label='predict Rf curve')
            xmajorLocator = MultipleLocator(0.1)
            ax.xaxis.set_major_locator(xmajorLocator)
            plt.legend(loc='lower right', prop=font1egend)
            plt.yticks(fontproperties='Arial', size=7)
            plt.xticks(fontproperties='Arial', size=7)
            plt.ylim(-0.1, 1.1)

            plt.savefig(f'PPT_fig/PE_Et2O_{int(index[a])}.tiff',
                        bbox_inches='tight', dpi=300)
            plt.savefig(f'PPT_fig/PE_Et2O_{int(index[a])}.pdf',
                        bbox_inches='tight', dpi=300)
            #plt.show()
            plt.cla()
            print(f'{j}:{smiles}')


    def test(self,X_test,y_test,model):
        '''
        Get test outcomes
        '''
        if self.use_model=='ANN':
            X_test = Variable(torch.from_numpy(X_test.astype(np.float32)).to(self.device), requires_grad=True)
            y_test = Variable(torch.from_numpy(y_test.astype(np.float32)).to(self.device))
            # X_importance=Variable(torch.zeros([1,X_test.shape[1]]).to(self.device),requires_grad=True)
            # y_importance=torch.autograd.grad(outputs=model(X_importance)[:, 0].sum(), inputs=X_importance, create_graph=True)[0]
            # print(y_importance)
            y_pred=model(X_test).cpu().data.numpy()
        elif self.use_model=='Ensemble':
            model_LGB,model_XGB,model_RF,model_ANN =model
            X_test_ANN = Variable(torch.from_numpy(X_test.astype(np.float32)).to(self.device), requires_grad=True)
            y_pred_ANN = model_ANN(X_test_ANN).cpu().data.numpy()
            y_pred_ANN=y_pred_ANN.reshape(y_pred_ANN.shape[0],)

            y_pred_XGB = model_XGB.predict(X_test)
            if self.use_sigmoid == True:
                y_pred_XGB = 1 / (1 + np.exp(-y_pred_XGB))

            y_pred_LGB = model_LGB.predict(X_test)
            if self.use_sigmoid == True:
                y_pred_LGB = 1 / (1 + np.exp(-y_pred_LGB))

            y_pred_RF = model_RF.predict(X_test)
            if self.use_sigmoid == True:
                y_pred_RF = 1 / (1 + np.exp(-y_pred_RF))

            #print(y_pred_LGB.shape,y_pred_XGB.shape,y_pred_ANN.shape)
            self.use_model='Ensemble'
            y_pred=(0.2*y_pred_LGB+0.2*y_pred_XGB+0.2*y_pred_RF+0.4*y_pred_ANN)

        else:
            y_pred=model.predict(X_test)
            if self.use_sigmoid==True:
                y_pred=1/(1+np.exp(-y_pred))

        #Model_ML.plot_predict_polarity(self, X_test,y_test,data_array,model)
        #Model_ML.plot_new_system(self, X_test,y_test,data_array,model)
        MSE,RMSE,MAE,R_square=Model_ML.plot_total_variance(self, y_test, y_pred)
        #plt.show()
        return y_pred,MSE,RMSE,MAE,R_square