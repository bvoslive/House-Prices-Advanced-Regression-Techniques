import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
#import warnings
#warnings.filterwarnings("ignore")

#IMPORTANDO

dataset = pd.read_csv('house_train.csv')
house_X2 = pd.read_csv('pre-processamento1.csv')

#IMPORTANDO SALEPRICE
saleprice = dataset['SalePrice']



#RANGE EM COLUNAS
house_X2.columns = range(len(house_X2.columns))



from math import log
saleprice = np.log(saleprice)






#excluindo variáveis com valor p alto
house_X2 = house_X2.drop([0, 2, 7, 8, 9, 12, 13, 15, 23], axis=1)


house_X2.columns = range(len(house_X2.columns))


house_X3 = house_X2.copy()

#LABEL ENCODER UNICAMENTE PARA PROCURAR CORRELAÇÕES
from sklearn.preprocessing import LabelEncoder

object_vetor = []

for i in range(len(house_X2.columns)):
    if(house_X2[i].describe().dtype=='object'):
        object_vetor.append(i)
        label_encoder = LabelEncoder()
        house_X2[i] = label_encoder.fit_transform(house_X2[i])

print('object_vetor', object_vetor)

print('indexes', house_X2.columns)

# PROCURANDO CORRELAÇÕES
corr2 = np.corrcoef(house_X2, rowvar=0)

eigenvalues, eigenvectors = np.linalg.eig(corr2)

vetor = []

# REMOVENDO VARIÁVEIS COLINEARES
for i in range(0, len(eigenvalues)):
    if (eigenvalues[i] < 1):
        vetor.append(i)

print('vetor', vetor)


"""
    CONTINUANDO
"""

#deletando variáveis colineares
house_X3 = house_X3.drop(vetor, axis=1)

print('vetor', vetor)

#variáveis para não deletar porque já foi deletado
#[15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]

#deletando variáveis com valor p alto
house_X3 = house_X3.drop([3,4,6,7,11,12,28,37], axis=1)


house_X3.columns = range(len(house_X3.columns))







#PROCURANDO INDEXES DE VARIÁVEIS DE NÍVEIS
for i in range(len(house_X3.columns)):
    if(house_X3[i].describe().dtype=='object'):
        print(i, '\n', house_X3[i].unique())


#deletando colunas com muitas dummy variables
house_X3 = house_X3.drop([3, 7, 8], axis=1)

house_X3.columns = range(len(house_X3.columns))
print('colunas de house3', house_X3.columns)


for i in range(len(house_X3.columns)):
    if(house_X3[i].describe().dtype=='object'):
        print(i, '\n', house_X3[i].unique())


"""
NÃO FAZER ONEHOTENCODER: 6,7,9
"""


#LABEL ENCODER DEFINITIVO
categorical_columns = []
for i in range(len(house_X3.columns)):
    if(house_X3[i].describe().dtype=='object'):
        categorical_columns.append(i)
        object_vetor.append(i)
        label_encoder = LabelEncoder()
        house_X3[i] = label_encoder.fit_transform(house_X3[i])


print('house_X3\n', house_X3)

print(categorical_columns)

from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features=[2])
house_X3 = onehotencoder.fit_transform(house_X3).toarray()


house_X3 = pd.DataFrame(house_X3)

house_X3.to_csv('pre-processamento2.csv')