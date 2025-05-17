import pymysql
import pandas as pd
from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
import mordred
import math
from rdkit.Chem import MACCSkeys
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

Eluent_smiles=['CCCCCC','CC(OCC)=O','C(Cl)Cl','CO','CCOCC']

class Dataset_process():
    '''
    For processing the data and split the dataset
    '''
    def __init__(self,config):
        super(Dataset_process, self).__init__()
        self.file_path=config.file_path
        self.dipole_path=config.dipole_path
        self.data_range=config.data_range
        self.choose_train=config.choose_train
        self.choose_validate=config.choose_validate
        self.choose_test=config.choose_test
        self.automatic_divide=config.automatic_divide
        self.seed=config.seed
        self.add_dipole=config.add_dipole
        self.add_molecular_descriptors=config.add_molecular_descriptors
        self.add_eluent_matrix = config.add_eluent_matrix
        self.add_MACCkeys=config.add_MACCkeys
        self.test_mode=config.test_mode
        self.download_data = config.download_data
        self.shuffle_array=config.shuffle_array
        if self.test_mode=='costum':
            self.costum_array = config.costum_array
        self.characterization_mode=config.characterization_mode

    def download_dataset(self,print_info=True):
        '''
        Download the dataset from mysql dataset
        :param print_info: whether print the download information
        :return: None
        '''
        dbconn = pymysql.connect(
            host='bj-cdb-k8stylt6.sql.tencentcdb.com',
            port=60474,
            user='xuhao',
            password='xuhao1101',
            database='TLC',
            charset='utf8',
        )

        # sql语句
        sqlcmd = "select * from tb_TLC"

        # 利用pandas 模块导入mysql数据
        a = pd.read_sql(sqlcmd, dbconn)

        a.to_excel(self.file_path)
        if print_info==True:
            print(f'Dataset has been downloaded, the file path is :{self.file_path}')

    def get_descriptor(self,smiles,ratio):
        compound_mol = Chem.MolFromSmiles(smiles)
        descriptor=[]
        descriptor.append(Descriptors.ExactMolWt(compound_mol))
        descriptor.append(Chem.rdMolDescriptors.CalcTPSA(compound_mol))
        descriptor.append(Descriptors.NumRotatableBonds(compound_mol))  # Number of rotable bonds
        descriptor.append(Descriptors.NumHDonors(compound_mol))  # Number of H bond donors
        descriptor.append(Descriptors.NumHAcceptors(compound_mol)) # Number of H bond acceptors
        descriptor.append(Descriptors.MolLogP(compound_mol)) # LogP
        descriptor=np.array(descriptor)*ratio
        return descriptor

    def get_eluent_descriptor(self,eluent_array):
        eluent=eluent_array
        des = np.zeros([6,])
        for i in range(eluent.shape[0]):
            if eluent[i] != 0:
                e_descriptors = Dataset_process.get_descriptor(self, Eluent_smiles[i], eluent[i])
                des+=e_descriptors
        return des

    def get_3D_conformer(self):
        data_range = self.data_range
        if self.download_data == True:
            Dataset_process.download_dataset(self)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info = entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]


        # 转变为mol并变成167维向量
        compound_mol = compound_smile.copy()

        use_index = 0
        for i in tqdm(range(len(compound_smile))):
            compound_mol[i] = Chem.MolFromSmiles(compound_smile[i])
            try:
                Finger = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(compound_smile[i]))
            except Exception as e:
                print(f'the compound {compound_ID[i]} has no MACCkeys.')
                continue
            mol = AllChem.MolFromSmiles(compound_smile[i])
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol)
            ### MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            Chem.MolToMolFile(new_mol, f'3D_conform/data_{i}.mol')

    def create_dataset(self,data_array,choose_num,compound_ID,dipole_ID,compound_Rf,compound_finger,compound_eluent,dipole_moment,
                       compound_MolWt,compound_TPSA,compound_nRotB,compound_HBD,compound_HBA,compound_LogP):
        '''
        create training/validate/test dataset
        add or not the molecular_descriptors and dipole moments can be controlled
        '''
        y = []
        database_finger = np.zeros([1, 167])
        database_eluent = np.zeros([1, 6])
        database_dipole = np.zeros([1, 1])
        database_descriptor = np.zeros([1, 6])
        for i in range(choose_num):
            index = int(data_array[i])
            ID_loc = np.where(compound_ID == index)[0]
            dipole_loc = np.where(dipole_ID == index)[0]
            for j in ID_loc:
                y.append([compound_Rf[j]])
                database_finger=np.vstack((database_finger,compound_finger[j]))
                database_eluent=np.vstack((database_eluent,compound_eluent[j]))
                if self.add_dipole==True:
                    database_dipole=np.vstack((database_dipole,dipole_moment[dipole_loc]))
                database_descriptor=np.vstack((database_descriptor,np.array([compound_MolWt[j],compound_TPSA[j],compound_nRotB[j],compound_HBD[j],compound_HBA[j],compound_LogP[j]]).reshape([1,6])))

        if self.add_MACCkeys==True:
            X=database_finger.copy()
            X=np.hstack((X,database_eluent))
        else:
            X=database_eluent.copy()

        if self.add_dipole==True:
            X=np.hstack((X,database_dipole))

        if self.add_molecular_descriptors == True:
            X =np.hstack((X,database_descriptor))


        X = np.delete(X, [0], axis=0)
        y = np.array(y)
        return X,y

    def delete_invalid(self,database, h):
        '''
        delete invalid data which is filled with -1 when reading the dataset
        '''
        delete_row_h = np.where(h == -1)[0]
        if delete_row_h.size > 0:
            database = np.delete(database, delete_row_h, axis=0)
            h = np.delete(h, delete_row_h, axis=0)

        delete_row_data = np.where(database == -1)[0]
        if delete_row_data.size > 0:
            database = np.delete(database, delete_row_data, axis=0)
            h = np.delete(h, delete_row_data, axis=0)
        return database,h

    def plot_compound(self,target_ID=-1):
        data_range = self.data_range
        if self.download_data == True:
            Dataset_process.download_dataset(self)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info = entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]
        compound_name=compound_info[:,10]
        compound_list = np.unique(compound_ID)
        compound_num = compound_list.shape[0]
        # data_array = compound_list.copy()
        # np.random.seed(self.seed)
        # np.random.shuffle(data_array)

        #----------------单个画图-----------------
        # index = target_ID
        # ID_loc = np.where(compound_ID == index)[0][0]
        # smile=compound_smile[ID_loc]
        # mol= Chem.MolFromSmiles(smile)
        # smiles_pic = Draw.MolToImage(mol, size=(500, 500),dpi=300, kekulize=True)
        # plt.figure(20,figsize=(0.5,0.5),dpi=300)
        # plt.imshow(smiles_pic)
        # plt.axis('off')
        # plt.savefig(f'fig_save/compound_{index}.tiff',dpi=300)
        # plt.savefig(f'fig_save/compound_{index}.pdf', dpi=300)
        # plt.show()


        #------------总体画图-----------------
        if target_ID==-1:
            plt.figure(10,figsize=(7,10),dpi=300)
            num=0
            for i in range(350,384):
                index=compound_list[i]
                ID_loc = np.where(compound_ID == index)[0][0]
                smile=compound_smile[ID_loc]
                mol= Chem.MolFromSmiles(smile)
                smiles_pic = Draw.MolToImage(mol, size=(200, 100), kekulize=True)
                plt.subplot(10,7,num+1)
                #plt.title(index)
                plt.imshow(smiles_pic)
                plt.axis('off')
                num+=1
            plt.savefig(f'fig_save/new/compound_{350}~{384}.tiff',dpi=300)
            plt.savefig(f'fig_save/new/compound_{350}~{384}.pdf',dpi=300)
            plt.show()

    def split_dataset(self):
        '''
        split the dataset according to the train/validate/test num
        :return: X_train,y_train,X_validate,y_validate,X_test,y_test,data_array(shuffled compounds)
        '''
        data_range=self.data_range
        if self.download_data==True:
            Dataset_process.download_dataset(self)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info=entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]
        compound_Rf = compound_info[:, 9]
        eluent = np.array(compound_info[:, 4:9],dtype=np.float32)
        compound_eluent=[]
        for j in range(eluent.shape[0]):
            des=Dataset_process.get_eluent_descriptor(self,eluent[j])
            compound_eluent.append(des.tolist())
        compound_eluent=np.array(compound_eluent)


        if self.add_eluent_matrix==False:
            Eluent_PE = compound_info[:, 4].copy()
            Eluent_EA = compound_info[:, 4].copy()
            for i in range(len(compound_eluent)):
                a = int(compound_eluent[i].split('-', )[1].split('_', )[1])
                b = int(compound_eluent[i].split('-', )[0].split('_', )[1])
                Eluent_PE[i] = b
                Eluent_EA[i] = a
                compound_eluent[i] = a / (a + b)

        # 转变为mol并变成167维向量
        compound_mol = compound_smile.copy()
        compound_finger = np.zeros([len(compound_smile), 167])
        compound_MolWt = np.zeros([len(compound_smile), 1])
        compound_TPSA = np.zeros([len(compound_smile), 1])
        compound_nRotB = np.zeros([len(compound_smile), 1])
        compound_HBD = np.zeros([len(compound_smile), 1])
        compound_HBA = np.zeros([len(compound_smile), 1])
        compound_LogP = np.zeros([len(compound_smile), 1])
        compound_ID_new = np.zeros([len(compound_smile), 1])
        compound_Rf_new = np.zeros([len(compound_smile), 1])
        compound_eluent_new = np.zeros([len(compound_smile), 6])

        use_index=0
        for i in tqdm(range(len(compound_smile))):
            compound_mol[i] = Chem.MolFromSmiles(compound_smile[i])
            try:
                Finger = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(compound_smile[i]))
            except Exception as e:
                print(f'the compound {compound_ID[i]} has no MACCkeys.')
                continue
            fingerprint = np.array([x for x in Finger])
            compound_finger[use_index] = fingerprint
            compound_MolWt[use_index] = Descriptors.ExactMolWt(compound_mol[i])
            if self.characterization_mode=='precise_TPSA':
                mol_conform=Chem.MolFromMolFile(f"3D_conform/data_{i}.mol")
                compound_TPSA[use_index] = mordred.CPSA.TPSA()(mol_conform)
            else:
                compound_TPSA[use_index] = Chem.rdMolDescriptors.CalcTPSA(compound_mol[i])
            compound_nRotB[use_index] = Descriptors.NumRotatableBonds(compound_mol[i])  # Number of rotable bonds
            compound_HBD[use_index] = Descriptors.NumHDonors(compound_mol[i])  # Number of H bond donors
            compound_HBA[use_index] = Descriptors.NumHAcceptors(compound_mol[i])  # Number of H bond acceptors
            compound_LogP[use_index] = Descriptors.MolLogP(compound_mol[i])  # LogP
            compound_ID_new[use_index]=compound_ID[i]
            compound_Rf_new[use_index]=compound_Rf[i]
            compound_eluent_new[use_index]=compound_eluent[i]
            use_index+=1


        compound_ID=compound_ID_new[0:use_index]
        compound_Rf=compound_Rf_new[0:use_index].reshape(compound_ID.shape[0],)
        compound_finger=compound_finger[0:use_index]
        compound_eluent=compound_eluent_new[0:use_index]
        compound_MolWt=compound_MolWt[0:use_index]
        compound_TPSA=compound_TPSA[0:use_index]
        compound_nRotB=compound_nRotB[0:use_index]
        compound_HBD=compound_HBD[0:use_index]
        compound_HBA=compound_HBA[0:use_index]
        compound_LogP=compound_LogP[0:use_index]

        # 读取偶极矩文件
        if self.add_dipole==True:
            dipole_info = (pd.read_excel(self.dipole_path, index_col=None, na_values=['NA']).fillna(-1)).values
            dipole_ID = dipole_info[:, 0]
            dipole_moment = dipole_info[:, 11]
        else:
            dipole_ID = None
            dipole_moment = None

        # 计算化合物的个数
        compound_list = np.unique(compound_ID)
        compound_num = compound_list.shape[0]
        # print(compound_num)
        if self.automatic_divide==True:
            self.choose_train=math.floor(0.8*compound_num)
            self.choose_validate=math.floor(0.1*compound_num)
            self.choose_test = math.floor(0.1 * compound_num)
        # print(self.choose_train,self.choose_validate,self.choose_test)
        if self.choose_train+self.choose_validate+self.choose_test>compound_num:
            raise ValueError(f'Out of compound num, which is {compound_num}')
        data_array = compound_list.copy()
        if self.shuffle_array==True:
            np.random.seed(self.seed)
            np.random.shuffle(data_array)

        X_train,y_train=Dataset_process.create_dataset(self,data_array[0:self.choose_train],self.choose_train,compound_ID, dipole_ID, compound_Rf, compound_finger,
                       compound_eluent, dipole_moment,compound_MolWt, compound_TPSA, compound_nRotB, compound_HBD, compound_HBA, compound_LogP)
        X_validate, y_validate = Dataset_process.create_dataset(self, data_array[self.choose_train:self.choose_train+self.choose_validate], self.choose_validate,
                                                          compound_ID, dipole_ID, compound_Rf, compound_finger,
                                                          compound_eluent, dipole_moment, compound_MolWt, compound_TPSA,
                                                          compound_nRotB, compound_HBD, compound_HBA, compound_LogP)
        if self.test_mode=='robot':
            X_test, y_test=Dataset_process.create_dataset(self, data_array[self.choose_train+self.choose_validate:self.choose_train+self.choose_validate+self.choose_test], self.choose_test,
                                                          compound_ID, dipole_ID, compound_Rf, compound_finger,
                                                          compound_eluent, dipole_moment, compound_MolWt, compound_TPSA,
                                                          compound_nRotB, compound_HBD, compound_HBA, compound_LogP)

        elif self.test_mode=='fix':
            X_test, y_test=Dataset_process.create_dataset(self, data_array[-self.choose_test-1:-1], self.choose_test,
                                                          compound_ID, dipole_ID, compound_Rf, compound_finger,
                                                          compound_eluent, dipole_moment, compound_MolWt, compound_TPSA,
                                                          compound_nRotB, compound_HBD, compound_HBA, compound_LogP)

        elif self.test_mode=='costum':
            X_test, y_test = Dataset_process.create_dataset(self, self.costum_array,
                                                            len(self.costum_array),
                                                            compound_ID, dipole_ID, compound_Rf, compound_finger,
                                                            compound_eluent, dipole_moment, compound_MolWt,
                                                            compound_TPSA,
                                                            compound_nRotB, compound_HBD, compound_HBA, compound_LogP)





        X_train,y_train=Dataset_process.delete_invalid(self,X_train,y_train)
        X_validate, y_validate = Dataset_process.delete_invalid(self, X_validate, y_validate)
        X_test,y_test=Dataset_process.delete_invalid(self, X_test, y_test)


        return X_train,y_train,X_validate,y_validate,X_test,y_test,data_array

    def split_dataset_all(self):
        '''
        split the dataset according to the TLC_num
        :return: X
        '''
        data_range = self.data_range
        if self.download_data == True:
            Dataset_process.download_dataset(self)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info = entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]
        compound_Rf = compound_info[:, 9]
        eluent = np.array(compound_info[:, 4:9], dtype=np.float32)
        compound_eluent = []
        for j in range(eluent.shape[0]):
            des = Dataset_process.get_eluent_descriptor(self, eluent[j])
            compound_eluent.append(des.tolist())
        compound_eluent = np.array(compound_eluent)
        if self.add_eluent_matrix == False:
            Eluent_PE = compound_info[:, 4].copy()
            Eluent_EA = compound_info[:, 4].copy()
            for i in range(len(compound_eluent)):
                a = int(compound_eluent[i].split('-', )[1].split('_', )[1])
                b = int(compound_eluent[i].split('-', )[0].split('_', )[1])
                Eluent_PE[i] = b
                Eluent_EA[i] = a
                compound_eluent[i] = a / (a + b)

        # 转变为mol并变成167维向量
        compound_mol = compound_smile.copy()
        compound_finger = np.zeros([len(compound_smile), 167])
        compound_MolWt = np.zeros([len(compound_smile), 1])
        compound_TPSA = np.zeros([len(compound_smile), 1])
        compound_nRotB = np.zeros([len(compound_smile), 1])
        compound_HBD = np.zeros([len(compound_smile), 1])
        compound_HBA = np.zeros([len(compound_smile), 1])
        compound_LogP = np.zeros([len(compound_smile), 1])
        compound_ID_new = np.zeros([len(compound_smile), 1])
        compound_Rf_new = np.zeros([len(compound_smile), 1])
        compound_eluent_new = np.zeros([len(compound_smile), 6])

        use_index = 0
        for i in range(len(compound_smile)):
            compound_mol[i] = Chem.MolFromSmiles(compound_smile[i])
            try:
                Finger = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(compound_smile[i]))
            except Exception as e:
                print(f'the compound {compound_ID[i]} has no MACCkeys.')
                continue
            fingerprint = np.array([x for x in Finger])
            compound_finger[use_index] = fingerprint
            compound_MolWt[use_index] = Descriptors.ExactMolWt(compound_mol[i])
            if self.characterization_mode=='precise_TPSA':
                mol_conform=Chem.MolFromMolFile(f"3D_conform/data_{i}.mol")
                compound_TPSA[use_index] = mordred.CPSA.TPSA()(mol_conform)
            else:
                compound_TPSA[use_index] = Chem.rdMolDescriptors.CalcTPSA(compound_mol[i])
            compound_nRotB[use_index] = Descriptors.NumRotatableBonds(compound_mol[i])  # Number of rotable bonds
            compound_HBD[use_index] = Descriptors.NumHDonors(compound_mol[i])  # Number of H bond donors
            compound_HBA[use_index] = Descriptors.NumHAcceptors(compound_mol[i])  # Number of H bond acceptors
            compound_LogP[use_index] = Descriptors.MolLogP(compound_mol[i])  # LogP
            compound_ID_new[use_index] = compound_ID[i]
            compound_Rf_new[use_index] = compound_Rf[i]
            compound_eluent_new[use_index] = compound_eluent[i]
            use_index += 1

        compound_ID = compound_ID_new[0:use_index]
        compound_Rf = compound_Rf_new[0:use_index].reshape(compound_ID.shape[0], )
        compound_finger = compound_finger[0:use_index]
        compound_eluent = compound_eluent_new[0:use_index]
        compound_MolWt = compound_MolWt[0:use_index]
        compound_TPSA = compound_TPSA[0:use_index]
        compound_nRotB = compound_nRotB[0:use_index]
        compound_HBD = compound_HBD[0:use_index]
        compound_HBA = compound_HBA[0:use_index]
        compound_LogP = compound_LogP[0:use_index]
        # 读取偶极矩文件
        if self.add_dipole == True:
            dipole_info = (pd.read_excel(self.dipole_path, index_col=None, na_values=['NA']).fillna(-1)).values
            dipole_ID = dipole_info[:, 0]
            dipole_moment = dipole_info[:, 11]
        else:
            dipole_ID = None
            dipole_moment = None

        y = []
        ID=[]
        database_finger = np.zeros([1, 167])
        database_eluent = np.zeros([1, 6])
        database_dipole = np.zeros([1, 1])
        database_descriptor = np.zeros([1, 6])
        for i in range(compound_finger.shape[0]):
            dipole_loc = np.where(dipole_ID == compound_ID[i])[0]
            y.append([compound_Rf[i]])
            ID.append([compound_ID[i]])
            database_finger = np.vstack((database_finger, compound_finger[i]))
            database_eluent = np.vstack((database_eluent, compound_eluent[i]))
            if self.add_dipole == True:
                database_dipole = np.vstack((database_dipole, dipole_moment[dipole_loc]))
            database_descriptor = np.vstack((database_descriptor, np.array(
                [compound_MolWt[i], compound_TPSA[i], compound_nRotB[i], compound_HBD[i], compound_HBA[i],
                 compound_LogP[i]]).reshape([1, 6])))

        if self.add_MACCkeys==True:
            X = database_finger.copy()
            X = np.hstack((X, database_eluent))
        else:
            X = database_eluent.copy()
        if self.add_dipole == True:
            X = np.hstack((X, database_dipole))
        if self.add_molecular_descriptors == True:
            X = np.hstack((X, database_descriptor))

        if self.characterization_mode=='no_multi':
            X = np.delete(X, [27,42,46,103], axis=1)
        X = np.delete(X, [0], axis=0)
        y = np.array(y)
        return X,y,ID
