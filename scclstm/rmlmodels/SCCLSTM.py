"""CLDNNLike model for RadioML.

# Reference:

- [CONVOLUTIONAL,LONG SHORT-TERM MEMORY, FULLY CONNECTED DEEP NEURAL NETWORKS ]

Adapted from code contributed by Mika.
"""
import os

WEIGHTS_PATH = ('resnet_like_weights_tf_dim_ordering_tf_kernels.h5')

from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,Dropout,Softmax,concatenate,Flatten,Reshape
from keras.layers.convolutional import Conv2D
from keras.layers import CuDNNLSTM


def ConvBNReluUnit(input,kernel_size = 8,index = 0):
    x = Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu',kernel_initializer='glorot_uniform',
               name='conv{}'.format(index + 1))(input)
    # x = BatchNormalization(name='conv{}-bn'.format(index + 1))(x)
    x = MaxPool1D(pool_size=2, strides=2, name='maxpool{}'.format(index + 1))(x)
    return x

def SCCLSTM(weights=None,
             input_shape1=[2,128],
             input_shape2=[1,128],
             classes=11,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate (%)
    tap = 8
    input1 =Input(input_shape1+[1],name='input1')
    input2 =Input(input_shape2+[1],name='input2')
    input3=Input(input_shape2+[1],name='input3')
    #Cnvolutional Block
    # L = 4
    # for i in range(L):
    #     x = ConvBNReluUnit(x,kernel_size = tap,index=i)

    # SeparateChannel Combined Convolutional Neural Networks
    x1=Conv2D(512,(2,18),padding='same', activation="relu", name="conv1_1", kernel_initializer='glorot_uniform')(input1)
    x2=Conv2D(256,(1,18),padding='same', activation="relu", name="conv1_2", kernel_initializer='glorot_uniform')(input2)
    x3=Conv2D(256,(1,18),padding='same', activation="relu", name="conv1_3", kernel_initializer='glorot_uniform')(input3)
    x=concatenate([x2,x3],axis=1)
    x=Conv2D(256,(1,9),padding='same', activation="relu", name="conv2", kernel_initializer='glorot_uniform')(x)
    x=concatenate([x1,x])
    x=Conv2D(128,(1,7),padding='valid', activation="relu", name="conv3", kernel_initializer='glorot_uniform')(x)
    x=Conv2D(64,(2,5),padding='valid', activation="relu", name="conv4", kernel_initializer='glorot_uniform')(x)
    #LSTM Unit
    # batch_size,64,2
    x= Reshape(target_shape=((118,64)),name='reshape')(x)
    x = CuDNNLSTM(units=64,return_sequences = True)(x)
    x = CuDNNLSTM(units=64)(x)

    #DNN
    x = Dense(128,activation='selu',name='fc1')(x)
    x=Dropout(dr)(x)
    x = Dense(128,activation='selu',name='fc2')(x)
    x=Dropout(dr)(x)
    x = Dense(classes,activation='softmax',name='softmax')(x)

    model = Model(inputs = [input1,input2,input3],outputs = x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

import keras
from keras.utils.vis_utils import plot_model
if __name__ == '__main__':
    model = SCCLSTM(None,classes=11)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    plot_model(model, to_file='model.png',show_shapes=True) # print model
    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())