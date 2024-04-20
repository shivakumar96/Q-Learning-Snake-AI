
import tensorflow as tf
from tensorflow import keras
from keras.optimizers.legacy import Adam
from keras.models import load_model
tf.compat.v1.disable_eager_execution()

HIDDEN_DIM_1 = 256
#HIDDEN_DIM_2 = 64
#HIDDEN_DIM_3 = 32

class NNModel:
    def __init__(self, lr, action_size, input_dim ):
        self.model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(HIDDEN_DIM_1,activation='relu'),
            #keras.layers.Dense(HIDDEN_DIM_2,activation='relu'),
            #keras.layers.Dense(HIDDEN_DIM_3,activation='relu'),
            keras.layers.Dense(action_size,activation='linear')
        ])
        self.model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        
    
    def saveModel()->None:
        pass
    

if __name__ == '__main__':
    cl =  NNModel(0.1,5,3)
    print(cl.model.summary())