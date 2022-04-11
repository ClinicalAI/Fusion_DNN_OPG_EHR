import random

from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, UpSampling2D
from keras.optimizer_v2.adam import Adam
from sklearn.decomposition import PCA
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Model
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Reshape
from keras.layers import core
from keras.layers import Dropout
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import ConvLSTM2D
import os
import keras.models as models
from skimage.transform import resize
from skimage.io import imsave
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
import keras
from keras.models import Model
from keras import layers
from keras.layers import Input, concatenate, Conv2D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from IPython.display import clear_output
import os,random
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.applications.densenet import DenseNet121,DenseNet169
from keras.applications.vgg16 import VGG16
from keras.applications.efficientnet import EfficientNetB0
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Input,Flatten,Dense,LeakyReLU,Dropout,GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def loading_data(img_npz_path, ehr_path):
    """**Load OPG images**"""
    image_data = np.load(img_npz_path)
    training_image = image_data['x']

    ## croping 
    train_data = training_image[:int(0.9*training_image.shape[0]),36:164,72:328]
    valid_data = training_image[int(0.9*training_image.shape[0]):,36:164,72:328]

    data = pd.read_csv(ehr_path)
    ids = np.array(data['code'])
    label = np.array(data['Stage'])

    ## selecting the labels
    label = np.floor(label/2)
    label_cat = np.reshape(label, (-1, 1))
    enc = OneHotEncoder()
    label_cat = enc.fit_transform(label_cat).toarray()
    train_label_cat = label_cat[:int(0.9*training_image.shape[0])]
    valid_label_cat = label_cat[int(0.9*training_image.shape[0]):]
    
    return train_data, valid_data, train_label_cat, valid_label_cat, ids


    
def Autoencoder(input_size = (128,256,1)):
    N = input_size[0]
    inputs = Input(input_size) 

    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization(axis=3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='drop3')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # D1
    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
    conv4_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4_1 = Dropout(0.5)(conv4_1)
    # D2
    conv4_2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
    conv4_2 = BatchNormalization(axis=3)(conv4_2)
    conv4_2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    conv4_2 = Dropout(0.5)(conv4_2)
    # D3
    merge_dense = concatenate([conv4_2,drop4_1], axis = 3)
    conv4_3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
    conv4_3 = BatchNormalization(axis=3)(conv4_3)
    conv4_3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='ae_out')(conv4_3)
    # drop4_3 = Dropout(0.5)(conv4_3)
    

    up6 = Conv2DTranspose(32, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal',name='up6')(conv4_3)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)
    x1 = Reshape(target_shape=(1, 32, 64, 32))(drop3)
    x2 = Reshape(target_shape=(1, 32, 64, 32))(up6)
    merge6  = concatenate([x1,x2], axis = 1) 
    merge6 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
            
    conv6 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up6)

    conv6 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(32, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)

    x1 = Reshape(target_shape=(1, 64, 128, 32))(conv2)
    x2 = Reshape(target_shape=(1, 64, 128, 32))(up7)
    merge7  = concatenate([x1,x2], axis = 1) 
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
        



    conv7 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(4, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)    
    
    conv8 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
    # conv8 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    # conv8 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv9 = Conv2D(1, 1,activation='sigmoid',name = 'desire')(conv8)


    # conv4_3_f_1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='ae_out2')(conv4_3)
    # conv4_3_f_2 = Conv2D(4, 3, activ  ation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3_f_1)
    # conv4_3_f_2 = MaxPooling2D(pool_size=(2, 2))(conv4_3_f_2)

    # conv4_3_f_3 = Conv2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='ae_out2')(conv4_3_f_2)

    # x = Flatten()(conv4_3)
    x = GlobalAveragePooling2D()(conv4_3)
    # x = Dense(units=128)(x)
    # x = LeakyReLU(alpha=0.3)(x)   
    x = Dense(units=32)(x)
    x = LeakyReLU(alpha=0.3)(x)  
    x = Dropout(0.5)(x)
    x = Dense(units=8)(x)
    x = LeakyReLU(alpha=0.3)(x)   
    outputs = Dense(units=2,activation='softmax')(x)
    model = Model(inputs=inputs, outputs=[outputs,conv9])
    return model


def trianing_ae_model(Autoencoder_Model, train_data, valid_data, train_label_cat, valid_label_cat, epochs=10):
    # Autoencoder_Model.load_weights('/content/drive/MyDrive/OPG_Classifier/models/unet_classifier.h5')
    for lr in [ 0.0001,  0.00001]:
        g_opt = Adam(learning_rate= lr)
        Autoencoder_Model.compile(optimizer=g_opt, loss=[tf.keras.losses.BinaryCrossentropy(),'mse'],metrics=['accuracy'])
        history = Autoencoder_Model.fit(
                    train_data,
                    [train_label_cat, train_data],
                    epochs=epochs,
                    batch_size=32,
                    validation_data=[valid_data,[valid_label_cat,valid_data]])





def laten_space_save(model, train_data, valid_data, train_label_cat, valid_label_cat, ids):
    latent_1 = Model(model.input,model.get_layer('ae_out').output) 
    la_train_1=np.reshape(latent_1.predict(train_data),(train_data.shape[0],16*32*16))
    la_valid_1=np.reshape(latent_1.predict(valid_data),(valid_data.shape[0],16*32*16))


    y_train = np.argmax(train_label_cat,axis=-1)
    y_test = np.argmax(valid_label_cat,axis=-1)

    tra_y_pred0,_ = model.predict(train_data)
    tra_y_pred = np.argmax(tra_y_pred0, axis=-1)

    val_y_pred0,_ = model.predict(valid_data)
    val_y_pred = np.argmax(val_y_pred0, axis=-1)

    latent_dic = {'Laten_data':np.append(la_train_1,la_valid_1,axis=0),
                'ids':ids,
                'laten_label_real':np.append(y_train,y_test,axis=0),
                'laten_label_pred':np.append(tra_y_pred,val_y_pred,axis=0)
                }
    if not os.path.exists('./latent_space'):
        os.makedirs('./latent_space')
    np.savez('./latent_space/latent_dic.npz',**latent_dic)
    return la_train_1, la_valid_1


def pca_vis(X_train, X_test, train_label_cat, valid_label_cat):

    y_train = np.argmax(train_label_cat,axis=-1)
    y_test = np.argmax(valid_label_cat,axis=-1)

    n_neighbors = 3
    random_state =  0

    X = np.append(X_train,X_test,axis=0)
    y = np.append(y_train,y_test,axis=0)

    dim = len(X[0])
    n_classes = len(np.unique(y))

    # Reduce dimension to 2 with PCA
    pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=random_state))


    # Make a list of the methods to be compared
    dim_reduction_methods = [("PCA", pca)]

    # plt.figure()
    for i, (name, model) in enumerate(dim_reduction_methods):
        plt.figure()
        # plt.subplot(1, 3, i + 1, aspect=1)

        # Fit the method's model
        model.fit(X_train, y_train)

        # Fit a nearest neighbor classifier on the embedded training set

        # Embed the data set in 2 dimensions using the fitted model
        X_embedded = model.transform(X)

        # Plot the projected points and show the evaluation score
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap="RdYlGn")
    plt.savefig('./latent_space/PCA.png')