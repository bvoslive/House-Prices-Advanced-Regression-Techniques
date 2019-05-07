import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
#import warnings
#warnings.filterwarnings("ignore")

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

"""

CRIAR MAP e substituir LabelEncoder por números, pq label encoder deixa na ordem alfabética

ONE HOT ENCODE: não fazer
LandSlope
ExterQual
ExterCond
BsmtQual
BsmtCond
BsmtExposure
BsmtFinType1
BsmtFinType2
HeatingQC
KitchenQual
FireplaceQu
GarageQual
GarageCond
PoolQC
Fence
OverallQual - números
OverallCond - números
"""

house_X = pd.read_csv('house_train.csv')
house_X2 = pd.read_csv('house_train.csv')
house_test = pd.read_csv('house_test.csv')

#VARIÁVEL SALEPRICE
saleprice = house_X2['SalePrice']


#deletando a coluna ID e saleprice
house_X2 = house_X2.drop('Id', axis=1)
house_X2 = house_X2.drop('SalePrice', axis=1)




#transformando colunas em números
colunas = house_X.columns
house_X.columns = range(len(house_X.columns))





#MAPS
MSZoning = {'C (all)': 'C', 'RM': 'RM', 'FV': 'FV', 'RH': 'RH'}
house_X2['MSZoning'] = house_X2['MSZoning'].map(MSZoning)

LandSlope = {'Gtl': '1Gtl', 'Mod': '2Mod', 'Sev': '3Sev'}
house_X2['LandSlope'] = house_X2['LandSlope'].map(LandSlope)

ExterQual = {'Ex':'5Ex', 'Gd':'4Gd', 'TA':'3TA', 'Fa':'2Fa', 'Po':'1Po'}
house_X2['ExterQual'] = house_X2['ExterQual'].map(ExterQual)

ExterCond = {'Ex':'5Ex', 'Gd':'4Gd', 'TA':'3TA', 'Fa':'2Fa', 'Po':'1Po'}
house_X2['ExterCond'] = house_X2['ExterCond'].map(ExterCond)

BsmtQual = {'Ex':'5Ex', 'Gd':'4Gd', 'TA':'3TA', 'Fa':'2Fa', 'Po':'1Po', 'NA':'0NA'}
house_X2['BsmtQual'] = house_X2['BsmtQual'].map(BsmtQual)

BsmtCond = {'Ex':'5Ex', 'Gd':'4Gd', 'TA':'3TA', 'Fa':'2Fa', 'Po':'1Po', 'NA':'0NA'}
house_X2['BsmtCond'] = house_X2['BsmtCond'].map(BsmtCond)

BsmtExposure = {'Gd':'4Gd', 'Av':'3Av', 'Mn':'2Mn', 'No':'1No', 'NA':'0NA'}
house_X2['BsmtExposure'] = house_X2['BsmtExposure'].map(BsmtExposure)

BsmtFinType1 = {'GLQ': '6GLQ', 'ALQ':'5ALQ','BLQ':'4BLQ','Rec':'3Rec','LwQ':'2LwQ','Unf':'1Unf','NA':'0NA'}
house_X2['BsmtFinType1'] = house_X2['BsmtFinType1'].map(BsmtFinType1)

BsmtFinType2 = {'GLQ': '6GLQ', 'ALQ':'5ALQ','BLQ':'4BLQ','Rec':'3Rec','LwQ':'2LwQ','Unf':'1Unf','NA':'0NA'}
house_X2['BsmtFinType2'] = house_X2['BsmtFinType2'].map(BsmtFinType2)

HeatingQC = {'Ex':'5Ex', 'Gd':'4Gd', 'TA':'3TA', 'Fa':'2Fa', 'Po':'1Po', 'NA':'0NA'}
house_X2['HeatingQC'] = house_X2['HeatingQC'].map(HeatingQC)

KitchenQual = {'Ex':'5Ex', 'Gd':'4Gd', 'TA':'3TA', 'Fa':'2Fa', 'Po':'1Po', 'NA':'0NA'}
house_X2['KitchenQual'] = house_X2['KitchenQual'].map(KitchenQual)

FireplaceQu = {'Ex':'5Ex', 'Gd':'4Gd', 'TA':'3TA', 'Fa':'2Fa', 'Po':'1Po', 'NA':'0NA'}
house_X2['FireplaceQu'] = house_X2['FireplaceQu'].map(FireplaceQu)

GarageQual = {'Ex':'5Ex', 'Gd':'4Gd', 'TA':'3TA', 'Fa':'2Fa', 'Po':'1Po', 'NA':'0NA'}
house_X2['GarageQual'] = house_X2['GarageQual'].map(GarageQual)

GarageCond = {'Ex':'5Ex', 'Gd':'4Gd', 'TA':'3TA', 'Fa':'2Fa', 'Po':'1Po', 'NA':'0NA'}
house_X2['GarageCond'] = house_X2['GarageCond'].map(GarageCond)

PoolQC = {'Ex':'4Ex', 'Gd':'3Gd', 'TA':'2TA', 'Fa':'1Fa', 'NA':'0NA'}
house_X2['PoolQC'] = house_X2['PoolQC'].map(PoolQC)

Fence = {'GdPrv':'4GdPrv', 'MnPrv':'3MnPrv', 'GdWo':'2GdWo', 'MnWw':'1MnWw', 'NA':'0NA'}
house_X2['Fence'] = house_X2['Fence'].map(Fence)

"""
#substituindo uma classe de uma variável problemática
for p in range(len(house_X2)):
    if(house_X2['MSZoning'][p]=='C (all)'):
        house_X2['MSZoning'][p]= 'C'
        house_test['MSZoning'][p] = 'C'
"""


house_X2.columns = range(len(house_X2.columns))
house_test.columns = range(len(house_test.columns))


#MOSTRANDO VARIÁVEIS COM MAIS DE 20% DE NULOS

for i in range(len(house_X2.columns)):
    if(house_X2[i].isnull().any()==True):
        if(house_X2[i].isnull().sum()/len(house_X2)>0.2):
            print('itens excluidos', i)
            house_X2 = house_X2.drop(i, axis=1)



#range nas colunas
house_X2.columns = range(len(house_X2.columns))



"""
#você parou aqui - DELETANDO OUTLIERS
lines = []

for i in range(len(house_X2.columns)):
    if (house_X2[i].describe().dtype == 'float'):
        quartil1 = house_X2[i].quantile(0.25)
        quartil3 = house_X2[i].quantile(0.75)
        for j in range(len(house_X2)):
            if((house_X2[i][j]<quartil1) or (house_X2[i][j]>quartil3)):
                print('quartis', quartil1, quartil3)
                print('number = ', j)
"""



#PREENCHENDO NULOS
for i in range(len(house_X2.columns)):
    if(house_X2[i].isnull().any()):
        if(house_X2[i].describe().dtype=='object'):
            unique_percent = pd.Series(house_X2[i].value_counts(1, dropna=True))
            #print(unique_percent.values)
            for j in range(len(house_X2)):
                if(pd.isnull(house_X2[i][j])==True):
                    house_X2[i][j] = np.random.choice(unique_percent.index, p=unique_percent.values)

        if(house_X2[i].describe().dtype=='float'):


            std = house_X2[i].std()
            mean = house_X2[i].mean()
            for j in range(len(house_X2)):
                if(pd.isnull(house_X2[i][j])==True):
                    house_X2[i][j] = np.random.randint(mean-std, mean+std)


house_X2.to_csv('pre-processamento1.csv', header=True)

