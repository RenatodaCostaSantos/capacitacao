
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Input





def criar_modelo_regularizado(neuronios_entrada, neurons=45, learning_rate=0.001, dropout_rate=0.3):
    """Cria e compila um modelo de Rede Neural com Dropout."""
    tf.random.set_seed(42)
    np.random.seed(42)

    tf.keras.backend.clear_session() # força o TensorFlow a liberar recursos de sessões antigas.
    
    optimizer = Adam(learning_rate=learning_rate)
    
    model = Sequential([
        # 1. Camada de Entrada

        Input(shape=(neuronios_entrada,)),
        
        Dense(neurons, activation='relu'),
        
        # 2. DROPOUT (Descarta 30% dos neurônios aleatoriamente)
        Dropout(dropout_rate), 
        
        # 3. Segunda Camada Oculta
        Dense(32, activation='relu'), 
        
        # 4. DROPOUT (Ajuda a evitar que a segunda camada decore)
        Dropout(dropout_rate), 
        
        # 5. Camada de Saída
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def cria_feature_media_ponderada(df):
    """
    Cria uma feature fazendo-se a média ponderada dos coeficientes do espectro de potencias da CMB.    
    """
    l_start = 2
    l_end = 511
   
    # Vetor l (o valor do momento de multipolo)
    l_values = np.arange(l_start, l_end + 1)

    # Vetor de pesos: l * (l + 1)
    weights = l_values * (l_values + 1) / 2*np.pi 

    # 4. Denominador da média ponderada (soma dos pesos)
    sum_of_weights = weights.sum()

    # 5. Cálculo do Numerador e da Nova Feature
    # O Pandas alinha as colunas de Cl_data com o vetor 'weights' e multiplica.
    # Depois, .sum(axis=1) soma o resultado dessa multiplicação ao longo das colunas,
    # resultando no numerador para CADA linha (observação).
    numerator = (df * weights).sum(axis=1)

    # Média ponderada
    df['media_ponderada_Cls'] = numerator / sum_of_weights

    return df