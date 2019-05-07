import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def dogao():

    """
    COLUNAS

    ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
           'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
           'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
           'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
           'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
           'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
           'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
           'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
           'SaleCondition', 'SalePrice'],
    """

    house_X = pd.read_csv('house_train.csv')
    house_X2 = pd.read_csv('house_train.csv')
    house_y = house_X['SalePrice']
    house_test = pd.read_csv('house_test.csv')

    house_catch = house_X

    colunas = house_X.columns

    house_X.columns = range(len(house_X.columns))

    """
    for i in range(len(colunas)):
        print(colunas[i], ' ', i, '\n')
        print(pd.unique(house_X[i]))
        print(house_X[i].value_counts(1, dropna=False))
        catch = house_X[i].describe()
        print('describe', catch.dtype)
        print('---------------')
    """

    # DELETANDO NULOS
    """
    house_X2 = house_X2.drop('Id', axis=1)
    house_X2 = house_X2.drop('Neighborhood', axis=1)
    house_X2 = house_X2.drop('MasVnrType', axis=1)
    house_X2 = house_X2.drop('MasVnrArea', axis=1)
    house_X2 = house_X2.drop('LotFrontage', axis=1)
    house_X2 = house_X2.drop('Alley', axis=1)
    house_X2 = house_X2.drop('SalePrice', axis=1)

    house_test = house_test.drop('Id', axis=1)
    house_test = house_test.drop('Neighborhood', axis=1)
    house_test = house_test.drop('MasVnrType', axis=1)
    house_test = house_test.drop('MasVnrArea', axis=1)
    house_test = house_test.drop('LotFrontage', axis=1)
    house_test = house_test.drop('Alley', axis=1)

    """
    # removendo C (all) de MSZoning
    house_X2 = house_X2[house_X2['MSZoning'] != 'C (all)']
    house_test = house_test[house_test['MSZoning'] != 'C (all)']

    # selecionando colunas
    house_X2.columns = range(len(house_X2.columns))
    house_test.columns = range(len(house_test.columns))

    for i in range(len(house_X2.columns)):
        catchnulls = house_X2[i].isnull()
        if (catchnulls.any() == True):
            house_X2[i].fillna(method='ffill', inplace=True)

    # house_X2[51] = house_X2[51].dropna(inplace=True)

    # você parou aqui

    # house_X2[66] = house_X2[66].fillna(method='ffill', inplace=True)

    # print(house_X2[66].value_counts(1, dropna=False))


    """
    #teste PROCURANDO POR NULOS
    
    cont_nulls = house_X2[66].isnull()
    cont_nulls = list(cont_nulls)
    if(cont_nulls.count(True)>0):
        uniques = house_X2[66].unique()[1:]
        random = np.random.choice(uniques)
        print('random', random)
        house_X2[66].fillna(random, inplace=True)
    
    """

    # house_X2[24].fillna(method='ffill', inplace=True)


    corr = house_X2.corr()

    # FAZENDO STANDARD SCALING E PREENCHENDO NULOS
    categorical_index = []
    categorical_column = []

    for i in range(len(house_X2.columns)):
        if (house_X2[i].describe().dtype == 'object'):

            categorical_index.append(i)

            # PROCURANDO E SUBSTITUINDO VALORES NULOS
            cont_nulls = house_X2[i].isnull()
            cont_nulls = list(cont_nulls)
            if (cont_nulls.count(True) > 0):
                uniques = house_X2[i].unique()[1:]
                random = np.random.choice(uniques)

                house_X2[i].fillna(random, inplace=True)

            label_encoder = LabelEncoder()

            house_X2[i] = label_encoder.fit_transform(house_X2[i])


    # PROCURANDO CORRELAÇÕES
    corr2 = np.corrcoef(house_X2, rowvar=0)
    eigenvalues, eigenvectors = np.linalg.eig(corr2)

    vetor = []

    # REMOVENDO VARIÁVEIS COLINEARES
    for i in range(0, len(eigenvalues)):
        if (eigenvalues[i] < 1):
            vetor.append(i)



    return vetor

print(dogao())

