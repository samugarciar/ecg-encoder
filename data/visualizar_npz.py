import numpy as np

data = np.load('processed/beats.npz')

for i in data.files:
    print('-' * 50)
    print(i)
    arrayActual = data[i]
    print(f'Valores: {arrayActual}')
    print(f'Forma: {arrayActual.shape}')
    print(f'Tipo de dato: {arrayActual.dtype}')


data.close()
