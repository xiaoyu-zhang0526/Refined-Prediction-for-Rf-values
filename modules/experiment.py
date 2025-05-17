import  os
import numpy as np
from xgboost.sklearn import XGBRegressor
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import time
import warnings
from sklearn.model_selection import GridSearchCV
from .dataset import *
from .models import *

warnings.filterwarnings("ignore")



class Conduct_Experiments(Model_ML):
    def __init__(self,config):
        super(Conduct_Experiments, self).__init__(config)

    def separation(self, X_test, y_test, y_pred, model):
        success_no_spapation=[]
        success_spapation = []
        failure_no_spapation=[]
        failure_spapation = []
        for i in range(X_test.shape[0]-1):
            for j in range(i,X_test.shape[0]):
                RE_i=np.abs(y_test[i]-y_pred[i])/y_test[i]
                RE_j = np.abs(y_test[j] - y_pred[j]) / y_test[j]
                if (X_test[i,167:173]==X_test[j,167:173]).all():
                    if y_test[i]<0.8 and y_test[i]>0.2:
                        if y_test[j]!=0:
                            if np.abs(y_test[i]-y_test[j])<=0.1 and np.abs(y_pred[i]-y_pred[j])<=0.1:
                                success_no_spapation.append(max(RE_i,RE_j))
                            elif np.abs(y_test[i]-y_test[j])>0.1 and np.abs(y_pred[i]-y_pred[j])>0.1:
                                success_spapation.append(max(RE_i,RE_j))
                            elif np.abs(y_test[i]-y_test[j])<=0.1 and np.abs(y_pred[i]-y_pred[j])>0.1:
                                failure_no_spapation.append(max(RE_i,RE_j))
                            elif np.abs(y_test[i]-y_test[j])>0.1 and np.abs(y_pred[i]-y_pred[j])<=0.1:
                                failure_spapation.append(max(RE_i,RE_j))
        success_spapation=np.array(success_spapation)
        success_no_spapation=np.array((success_no_spapation))
        failure_spapation=np.array(failure_spapation)
        failure_no_spapation=np.array(failure_no_spapation)
        print(f'successfully predict separation: {success_spapation.shape[0]}, mean relative error: {np.mean(success_spapation)}\n'
              f'successfully predict cannot separation: {success_no_spapation.shape[0]}, mean relative error: {np.mean(success_no_spapation)}\n'
              f'failure to predict separation: {failure_spapation.shape[0]}, mean relative error: {np.mean(failure_spapation)}\n'
              f'failure to predict cannot separation:{failure_no_spapation.shape[0]}, mean relative error: {np.mean(failure_no_spapation)}\n')
        print(min(failure_spapation),min(failure_no_spapation),np.percentile(failure_spapation,50),np.percentile(failure_no_spapation,50))

    def grid_search(self,X_train, y_train, X_validate, y_validate):
        '''
                train model using LightGBM,Xgboost,Random forest or ANN
                '''
        print('----------Start Training!--------------')

        torch.manual_seed(self.torch_seed)
        if self.use_model == 'LGB':
            parameters = {
                'max_depth': [1, 3, 5],
                'num_leaves': [5, 15, 25],
                'learning_rate': [0.0001, 0.0005, 0.0007]
            }
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model = lgb.LGBMRegressor(objective='regression', max_depth=self.LGB_max_depth,
                                      num_leaves=self.LGB_num_leaves,
                                      learning_rate=self.LGB_learning_rate, n_estimators=self.LGB_n_estimators)
            gsearch = GridSearchCV(model, param_grid=parameters, scoring='neg_mean_squared_error', cv=3)
            gsearch.fit(X_train, list(y_train.reshape(y_train.shape[0], )),
                      eval_set=[(X_train, list(y_train.reshape(y_train.shape[0], ))),
                                (X_validate, list(y_validate.reshape(y_validate.shape[0], )))],
                      eval_names=('fit', 'val'), eval_metric='l2', early_stopping_rounds=self.LGB_early_stopping_rounds,
                      verbose=False)
            print('参数的最佳取值:{0}'.format(gsearch.best_params_))
            print('最佳模型得分:{0}'.format(gsearch.best_score_))
            print(gsearch.cv_results_['mean_test_score'])
            print(gsearch.cv_results_['params'])
            print('----------LGB Training Finished!--------------')
            return model

        elif self.use_model == 'XGB':
            parameters = {
                'max_depth': [1,3,5],
                'n_estimators': [200, 300, 400],
                'learning_rate': [0.01, 0.1,1]
            }
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

            gsearch = GridSearchCV(model, param_grid=parameters, scoring='neg_mean_squared_error', cv=3)
            gsearch.fit(X_train, y_train.reshape(y_train.shape[0]))
            print('参数的最佳取值:{0}'.format(gsearch.best_params_))
            print('最佳模型得分:{0}'.format(gsearch.best_score_))
            print(gsearch.cv_results_['mean_test_score'])
            print(gsearch.cv_results_['params'])
            print('----------XGB Training Finished!--------------')
            return model

        elif self.use_model == 'RF':
            parameters = {
                'n_estimators': [1000, 2000, 3000],
            }
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model = RandomForestRegressor(n_estimators=self.RF_n_estimators,
                                          criterion='squared_error',
                                          random_state=self.RF_random_state,
                                          n_jobs=self.RF_n_jobs)
            gsearch = GridSearchCV(model, param_grid=parameters, scoring='neg_mean_squared_error', cv=3)
            gsearch.fit(X_train, y_train)
            print('参数的最佳取值:{0}'.format(gsearch.best_params_))
            print('最佳模型得分:{0}'.format(gsearch.best_score_))
            print(gsearch.cv_results_['mean_test_score'])
            print(gsearch.cv_results_['params'])
            print('----------RF Training Finished!--------------')
            return model


        elif self.use_model == 'Ensemble':
            y_train_origin = y_train.copy()
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
                          eval_names=('fit', 'val'), eval_metric='l2',
                          early_stopping_rounds=self.LGB_early_stopping_rounds,
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
                    dprediction = \
                        torch.autograd.grad(outputs=prediction[:, 0].sum(), inputs=X_train, create_graph=True)[0]
                    dprediction_deluent = dprediction[:, 167].reshape(X_train.shape[0], 1)
                    dprediction_validate = \
                        torch.autograd.grad(outputs=prediction_validate[:, 0].sum(), inputs=X_validate,
                                            create_graph=True)[0]
                    dprediction_validate_deluent = dprediction_validate[:, 167].reshape(X_validate.shape[0], 1)
                    MSELoss = torch.nn.MSELoss()
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


            model_ANN=Net
            X_valid_ANN = X_validate
            y_pred_ANN = model_ANN(X_valid_ANN).cpu().data.numpy()
            y_pred_ANN = y_pred_ANN.reshape(y_pred_ANN.shape[0], )
            X_validate=X_validate.cpu().data.numpy()
            y_validate = y_validate.cpu().data.numpy()

            y_pred_XGB = model_XGB.predict(X_validate)
            if self.use_sigmoid == True:
                y_pred_XGB = 1 / (1 + np.exp(-y_pred_XGB))

            y_pred_LGB = model_LGB.predict(X_validate)
            if self.use_sigmoid == True:
                y_pred_LGB = 1 / (1 + np.exp(-y_pred_LGB))

            y_pred_RF = model_RF.predict(X_validate)
            if self.use_sigmoid == True:
                y_pred_RF = 1 / (1 + np.exp(-y_pred_RF))

            weight=[0,0.1,0.2,0.3]
            for w in weight:
                y_pred = ( w* y_pred_LGB + w * y_pred_XGB + w * y_pred_RF + (1-3*w) * y_pred_ANN)

                MSE, RMSE, MAE, R_square = Model_ML.plot_total_variance(self, y_validate, y_pred)
                print(w,MSE, RMSE, MAE, R_square)

    def influence_of_weights(self,X_train, y_train, X_validate, y_validate,X_test,y_test):
        torch.manual_seed(self.torch_seed)
        y_train_origin = y_train.copy()
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
                      eval_names=('fit', 'val'), eval_metric='l2',
                      early_stopping_rounds=self.LGB_early_stopping_rounds,
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
                dprediction = \
                    torch.autograd.grad(outputs=prediction[:, 0].sum(), inputs=X_train, create_graph=True)[0]
                dprediction_deluent = dprediction[:, 167].reshape(X_train.shape[0], 1)
                dprediction_validate = \
                    torch.autograd.grad(outputs=prediction_validate[:, 0].sum(), inputs=X_validate,
                                        create_graph=True)[0]
                dprediction_validate_deluent = dprediction_validate[:, 167].reshape(X_validate.shape[0], 1)
                MSELoss = torch.nn.MSELoss()
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

        model_ANN = Net
        X_test_ANN = Variable(torch.from_numpy(X_test.astype(np.float32)).to(self.device), requires_grad=True)
        y_pred_ANN = model_ANN(X_test_ANN).cpu().data.numpy()
        y_pred_ANN = y_pred_ANN.reshape(y_pred_ANN.shape[0], )

        y_pred_XGB = model_XGB.predict(X_test)
        if self.use_sigmoid == True:
            y_pred_XGB = 1 / (1 + np.exp(-y_pred_XGB))

        y_pred_LGB = model_LGB.predict(X_test)
        if self.use_sigmoid == True:
            y_pred_LGB = 1 / (1 + np.exp(-y_pred_LGB))

        y_pred_RF = model_RF.predict(X_test)
        if self.use_sigmoid == True:
            y_pred_RF = 1 / (1 + np.exp(-y_pred_RF))


        weight = [0,0.05, 0.1,0.15, 0.2,0.25, 0.3]
        for w in weight:
            y_pred = (w * y_pred_LGB + w * y_pred_XGB + w * y_pred_RF + (1 - 3 * w) * y_pred_ANN)

            MSE, RMSE, MAE, R_square = Model_ML.plot_total_variance(self, y_test, y_pred)
            print(w, MSE, RMSE, MAE, R_square)
