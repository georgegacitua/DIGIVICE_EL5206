import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#%%
#Histograms
def histogram(data, xlabel):
    """
    Generates the histogram for the data

    :param data:    The data to plot
    """
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.clf()
    n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel('Frecuencia')
    plt.title('Histograma de {}'.format(xlabel))
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

#%%
#Data Pre-Process
def preprocess(data):
    """
    Does the preprocessing of the data

    :param data:        Data to preprocess
    :return:            The preprocessed data
    """
    # Change the gender data to numerical
    data['GENERO'] = data['GENERO'].apply(lambda x: 1 if ('F' in str(x)) else (0 if ('M' in str(x)) else -1))

    # Set the educational level in one hot codification
    data['MED'] = data['NIV_EDUC'].apply(lambda x: 1 if ('MED' in str(x)) else 0)
    data['TEC'] = data['NIV_EDUC'].apply(lambda x: 1 if ('TEC' in str(x)) else 0)
    data['UNV'] = data['NIV_EDUC'].apply(lambda x: 1 if ('UNV' in str(x)) else 0)
    data.drop('NIV_EDUC', axis=1, inplace=True)

    # Set the Civil State as one hot
    data['CAS'] = data['E_CIVIL'].apply(lambda x: 1 if ('CAS' in str(x)) else 0)
    data['SEP'] = data['E_CIVIL'].apply(lambda x: 1 if ('SEP' in str(x)) else 0)
    data['SOL'] = data['E_CIVIL'].apply(lambda x: 1 if ('SOL' in str(x)) else 0)
    data['VIU'] = data['E_CIVIL'].apply(lambda x: 1 if ('VIU' in str(x)) else 0)
    data.drop('E_CIVIL', axis=1, inplace=True)

    # Drop the origin city (redundant data)
    data.drop('CIUDAD', axis=1, inplace=True)

    # Change the AVAL to numerical
    data['Aval'] = data['Aval'].apply(lambda x: 1 if ('SI' in str(x)) else (0 if ('NO' in str(x)) else -1))
    data['PAGA'] = data['PAGA'].apply(lambda x: 0 if ('NO PAGA' in str(x)) else (1 if ('PAGA' in str(x)) else -1))

    return data

#%%
#Normalization
def normalize_column(data, *args):
    """
    Normalize the columns of a data frame

    :param data:    The data to normalize
    :param args:    The names of the columns to normalize
    :return:        The normalized data
    """
    for name in args:
        data[name] = (data[name] - data[name].mean()) / data[name].std()

    return data

#%%
# Names of the files
raw_name = 'CREDITRISK_RAW.xlsx'
score_name = 'CREDITRISK_SCORE.xlsx'

# Get the data
raw_data = pd.read_excel(raw_name)
score_data = pd.read_excel(score_name)

"""
Simple Preprocessing
"""
# Drop values for missing rows
raw_data = raw_data.replace('', np.NaN).dropna()
raw_data = raw_data.replace(' ', np.NaN).dropna()

# Delete columns

raw_data = raw_data.iloc[:, 1:]
score_data = score_data.iloc[:, 1:]

# Do preprocessing
raw_data = preprocess(raw_data)

# Obtain the results
results = raw_data['PAGA'].reset_index().drop('index', axis=1)
raw_data.drop('PAGA', axis=1, inplace=True)

"""
Make histograms of the variables
"""
histogram(raw_data['RENTA'], 'Renta')
histogram(raw_data['EDAD'], 'Edad')
histogram(raw_data['COD_OFI'], 'Codigo de Oficina')
histogram(raw_data['COD_COM'], 'Codigo de Comuna')
histogram(raw_data['Monto Deuda Promedio'], 'Deuda Promedio')
histogram(raw_data['Días de Mora'], 'Dias Mora')
histogram(raw_data['Monto solicitado'], 'Monto Solicitado')

"""
Normalize some columns
"""
raw_data = normalize_column(raw_data, 'RENTA', 'EDAD', 'Monto Deuda Promedio', 'Monto solicitado',
                            'Crédito_1', 'Crédito_2', 'Crédito_3', 'Crédito_4', 'Número de meses inactivo',
                            'numero de cuotas')

"""
Do a principal component analysis (PCA)
"""

pca = PCA(n_components=2)
pca.fit(raw_data)
pca_raw_data = pd.DataFrame(data=pca.fit_transform(raw_data))
pagan = pca_raw_data[results['PAGA'] == 1]
no_pagan = pca_raw_data[results['PAGA'] == 0]
print(pca.explained_variance_ratio_)

plt.clf()
plt.plot(pagan.iloc[:, 0], pagan.iloc[:, 1], 'r.', label='Personas que pagan')
plt.plot(no_pagan.iloc[:, 0], no_pagan.iloc[:, 1], 'b.', label='Personas que no pagan')
plt.xlabel('First PC')
plt.ylabel('Second PC')
plt.legend()
plt.grid()
plt.show()
print('Done')

msk = np.random.rand(len(raw_data)) < .6

train = raw_data[msk]
test = raw_data[~msk]
labels_train = results[msk]
labels_test = results[~msk]

msk_test = np.random.rand(len(test)) < .5

validation = test[~msk_test]

labels_validation = labels_test[~msk_test]

labels_test = labels_test[msk_test]
test = test[msk_test]
clf = SVC(gamma='auto')
clf.fit(train, labels_train)

predictions = clf.predict(validation)

print(classification_report(labels_validation, predictions))
