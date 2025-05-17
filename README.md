# 一、简介

本程序 `Rf_prediction.py` 是为论文《High-throughput discovery of chemical structure-polarity relationships combining automation and machine learning techniques》开发的核心代码。其主要功能是通过自动化和机器学习方法，预测化合物在薄层色谱（TLC）中的Rf值，实现高通量的结构-极性关系发现。

程序集成了数据处理、特征提取、模型训练（支持LightGBM、XGBoost、随机森林、神经网络、贝叶斯回归及集成学习）、模型评估和可视化等功能，适用于有机化学、药物化学等领域的高通量极性预测。

## 二、使用说明

### 1. 安装依赖

- Python 3.12
- 主要依赖包：`rdkit`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch`, `matplotlib`, `tqdm`, `PIL` 等
- 详见 `requirements.txt`

```shel
pip install -r requirements.txt
```

### 2. 训练

运行如下命令对模型进行训练，参数可省略，省略即使用默认值。

```bash
python train_and_test.py [参数]
```

### 3.预测Smile式的RF值

~~~shell
python redict_rf.py --smile_str [smile式] --dipole_num [dipole数] [其他参数]
~~~

例子：

~~~shell
python predict_rf.py \
--smile_str CC(OC[C@H]1O[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H](OC(C)=O)[C@H]1OC(C)=O)=O \
--dipole_num 6.66
~~~
