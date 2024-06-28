import os
from tensorflow import keras

# from keras.models import Model
from keras import models
# from keras.layers import Input,Dense,Conv1D,Dropout,concatenate,Reshape
from keras import layers
#from keras.Regulizers import L2
L1 = keras.regularizers.L1
L2 = keras.regularizers.L2
# from keras.layers import CuDNNLSTM

def MCLDNN(weights=None,
             input_shape1=[2,128],
             input_shape2=[128,1],
             classes=11,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr=0.5

    input1=layers.Input(input_shape1+[1],name='IQchannel')
    input2=layers.Input(input_shape2,name='Ichannel')
    input3=layers.Input(input_shape2,name='Qchannel')

    # Part-A: Multi-channel Inputs and Spatial Characteristics Mapping Section
    x1=layers.Conv2D(50,(2,8),padding='same',activation="relu",name="Conv1",kernel_initializer="glorot_uniform")(input1)
    x2=layers.Conv1D(50,8,padding='causal',activation="relu",name="Conv2",kernel_initializer="glorot_uniform")(input2)
    x2_reshape=layers.Reshape([-1,128,50])(x2)
    x3=layers.Conv1D(50,8,padding='causal',activation="relu",name="Conv3",kernel_initializer="glorot_uniform")(input3)
    x3_reshape=layers.Reshape([-1,128,50],name="reshap2")(x3)
    x=layers.concatenate([x2_reshape,x3_reshape],axis=1,name='Concatenate1')
    x=layers.Conv2D(50,(1,8), padding='same',activation="relu",name="Conv4",kernel_initializer="glorot_uniform")(x)
    x=layers.concatenate([x1,x],name="Concatenate2")
    x=layers.Conv2D(100,(2,5),padding="valid",activation="relu",name="Conv5",kernel_initializer="glorot_uniform")(x)
    #x=layers.Conv2D(100,(2,5),padding="valid",activation="relu",kernel_regularizer=L1(0.0001),name="Conv5_reg",kernel_initializer="glorot_uniform")(x)

    # Part-B: Temporal Characteristics Extraction Section
    x=layers.Reshape(target_shape=(124,100))(x)
    x=layers.LSTM(units=128,return_sequences=True,name="LSTM1")(x)
    x=layers.LSTM(units=128,name="LSTM2")(x)

    #DNN
    x=layers.Dense(128,activation="selu",name="FC1")(x)
    #x=layers.Dense(128, activation="selu", kernel_regularizer=L1(0.0001), name="FC1_reg")(x)
    x=layers.Dropout(dr)(x)
    
    x=layers.Dense(128,activation="selu",name="FC2")(x)
    #x=layers.Dense(128, activation="selu", kernel_regularizer=L2(0.0001), name="FC2_reg")(x)
    x=layers.Dropout(dr)(x)
    
    x=layers.Dense(classes,activation="softmax",name="Softmax")(x)

    model=models.Model(inputs=[input1,input2,input3],outputs=x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)
    
    return model

import keras
#from keras.optimizers import adam
#from keras import optimizers
if __name__ == '__main__':
    # for the RaioML2016.10a dataset
    model = MCLDNN(classes=11)

    # for the RadioML2016.10b dataset
    # model = MCLDNN(classes=10)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()