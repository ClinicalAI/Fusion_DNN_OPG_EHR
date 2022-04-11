import random
import os

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.model_selection import train_test_split


# Set seed numbers
SEED = 100
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Fusion model architecture
def get_model_EHR_OPG():
    #77
    inputs1 = layers.Input(8192)
    x = layers.Dense(units=256, activation="relu")(inputs1)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    f1 = layers.Dense(units=64, activation="relu")(x)


    inputs2 = layers.Input(8)

    x = layers.Dense(units=8, activation="relu")(inputs2)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=8, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=8, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    f2 = layers.Dense(units=8, activation="relu")(x)

    merge = layers.concatenate([f1,f2])
    x = layers.Dense(units=32, activation="relu")(merge)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=16, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=8, activation="relu")(x)

    outputs = layers.Dense(1, activation='sigmoid', name="classifier")(x)
    model = models.Model(inputs=[inputs1, inputs2], outputs=outputs, name="disease_prediction")

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

# Model for only OPG data
def get_model_OPG():
    inputs1 = layers.Input(8192)
    x = layers.Dense(units=256, activation="relu")(inputs1)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=16, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs1 , outputs=outputs, name="disease_prediction")

    model.compile(loss='binary_crossentropy',
                  optimizer="Adam",
                  metrics=['accuracy'])
    print(model.summary())
    return model

# Model for only EHR data
def get_model_EHR():
    inputs1 = layers.Input(8)

    x = layers.Dense(units=8, activation="relu")(inputs1)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=4, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=2, activation="relu")(x)
    # x = layers.Dense(units=2, activation="relu")(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs1, outputs=outputs, name="disease_prediction")
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model


"""**Train fusion model**"""
def train_model_EHR_OPG(epoch, _batch, combine_data, label, model_dir, ROC_dir, prefix='', col=''):
    current_dir = model_dir + os.sep + col
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)

    _epoch = epoch
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits)
    i = 1
    predictions = []
    con_mat_avg = []
    fprs = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    x = combine_data
    y = label


    x, y = shuffle(x, y, random_state=SEED)

    class_w = dict(zip(np.unique(label), class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(label), y=label)))

    for train_index, test_index in kf.split(x, y):
        y_train, y_test = y[train_index], y[test_index]

        ch1_train = np.stack(x[train_index, 0])
        ch2_train = np.stack(x[train_index, 1])
        ch1_test = np.stack(x[test_index, 0])
        ch2_test = np.stack(x[test_index, 1])


        model = get_model_EHR_OPG()

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            current_dir + os.sep + prefix + '_epoch_{}_{}.h5'.format(_epoch, i),
            monitor='val_accuracy', verbose=1,
            save_best_only=True, mode='max')
        log = tf.keras.callbacks.CSVLogger(current_dir + os.sep + prefix + '_epoch_{}_{}.log'.format(_epoch, i))

        print('\n')
        print('\n')
        print('**************************************************')
        print('training on fold {} out of {}'.format(i, n_splits))
        print('**************************************************')
        print('\n')
        print('\n')

        history = model.fit(
            [ch1_train, ch2_train],
            y_train,
            # verbose=0,
            epochs=_epoch,
            batch_size=_batch,
            validation_data=([ch1_test, ch2_test],y_test),
            callbacks=[checkpoint,
                       log

                       ],
        )

        model.load_weights(current_dir + os.sep + prefix + '_epoch_{}_{}.h5'.format(_epoch, i))
        pred_score = model.predict([ch1_test, ch2_test])
        pred_ = (pred_score > 0.5)       
        predictions.append(pred_)

        plt.figure()
        plt.plot(history.history['loss'], label='Binary crossentropy (training data)')
        plt.plot(history.history['val_loss'], label='Binart crossentropy (validation data)')
        plt.plot(history.history['accuracy'], label='Accuracy (training data)')
        plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
        plt.title('Model performance for 3D MNIST Keras Conv3D example')
        plt.ylabel('Loss value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.savefig(current_dir + os.sep + prefix + '_fold_{}.png'.format(i), dpi=600, bbox_inches='tight')

        con_mat = tf.math.confusion_matrix(labels=y_test, predictions=pred_).numpy()
        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        con_mat_avg.append(con_mat_norm)

        labels = ["0", "1"]

        con_mat_df = pd.DataFrame(con_mat_norm, index=labels, columns=labels)

        plt.figure()
        plt.rcParams.update({'font.size': 18})
        sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('confusion matrix fold {}'.format(i))
        plt.savefig(current_dir + os.sep + prefix + '_fold_{}.png'.format(i), dpi=600, bbox_inches='tight')
        #plt.show()

        fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(y_test, pred_score)
        tprs.append(np.interp(mean_fpr, fpr_keras, tpr_keras))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr_keras, tpr_keras)
        aucs.append(roc_auc)

        i += 1

    con_mat_avg = np.array(con_mat_avg)

    con_mat_df = pd.DataFrame(con_mat_avg.mean(axis=0), index=labels, columns=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix ICD-10 ({})'.format(col))
    plt.savefig(ROC_dir + os.sep + prefix + '_confusion_{}.png'.format(col), dpi=600,bbox_inches='tight')

    plt.figure(figsize=(10, 8))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    np.savez_compressed(ROC_dir + os.sep + '{}_ROC_info_{}.npz'.format(prefix,col), tprs=tprs, mean_fpr=mean_fpr,
                        mean_tpr=mean_tpr, aucs=aucs)
    plt.plot(mean_fpr, mean_tpr,
             label=r'(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=1.5, alpha=.9)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_dir + os.sep + prefix + '_ROC_{}.png'.format(col), dpi=600, bbox_inches='tight')



"""**Train model for only OPG features**"""
def train_model_OPG(epoch, _batch, combine_data, label, model_dir, ROC_dir, prefix='', col=''):
    current_dir = model_dir + os.sep + col
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)

    _epoch = epoch
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits)
    i = 1
    predictions = []
    con_mat_avg = []
    fprs = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    x = combine_data
    y = label

    x, y = shuffle(x, y, random_state=SEED)

    class_w = dict(zip(np.unique(label), class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(label), y=label)))

    for train_index, test_index in kf.split(x, y):
        y_train, y_test = y[train_index], y[test_index]
        ch1_train = np.stack(x[train_index, 0])       
        ch1_test = np.stack(x[test_index, 0])
        
        model = get_model_OPG()



        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            current_dir + os.sep + prefix + '_epoch_{}_{}.h5'.format(_epoch, i),
            monitor='val_accuracy', verbose=1,
            save_best_only=True, mode='max')
        log = tf.keras.callbacks.CSVLogger(current_dir + os.sep + prefix + '_epoch_{}_{}.log'.format(_epoch, i))

        print('\n')
        print('\n')
        print('**************************************************')
        print('training on fold {} out of {}'.format(i, n_splits))
        print('**************************************************')
        print('\n')
        print('\n')

        history = model.fit(
            ch1_train,
            y_train,
            # verbose=0,
            epochs=_epoch,
            batch_size=_batch,
            validation_data=(ch1_test,y_test),
            callbacks=[checkpoint,
                       log
                       ],
        )

        model.load_weights(current_dir + os.sep + prefix + '_epoch_{}_{}.h5'.format(_epoch, i))
        pred_score = model.predict(ch1_test)
        pred_ = (pred_score > 0.5)     
        predictions.append(pred_)

        plt.figure(figsize=(15, 8))
        plt.plot(history.history['loss'], label='Binary crossentropy (training data)')
        plt.plot(history.history['val_loss'], label='Binart crossentropy (validation data)')
        plt.plot(history.history['accuracy'], label='Accuracy (training data)')
        plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
        plt.title('Model performance')
        plt.ylabel('Loss value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.savefig(current_dir + os.sep + prefix + '_PER_{}.png'.format(i), dpi=600, bbox_inches='tight')

        con_mat = tf.math.confusion_matrix(labels=y_test, predictions=pred_).numpy()
        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        con_mat_avg.append(con_mat_norm)

        labels = ["0", "1"]

        con_mat_df = pd.DataFrame(con_mat_norm, index=labels, columns=labels)

        plt.figure()
        plt.rcParams.update({'font.size': 18})
        sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('confusion matrix fold {}'.format(i))
        plt.savefig(current_dir + os.sep + prefix + '_CON_{}.png'.format(i), dpi=600, bbox_inches='tight')

        fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(y_test, pred_score)
        tprs.append(np.interp(mean_fpr, fpr_keras, tpr_keras))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr_keras, tpr_keras)
        aucs.append(roc_auc)

        i += 1

    con_mat_avg = np.array(con_mat_avg)

    con_mat_df = pd.DataFrame(con_mat_avg.mean(axis=0), index=labels, columns=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix ICD-10 ({})'.format(col))
    plt.savefig(ROC_dir + os.sep + prefix + '_confusion_{}.png'.format(col), dpi=600,bbox_inches='tight')
    # n_classes = 2

    plt.figure(figsize=(10, 8))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    np.savez_compressed(ROC_dir + os.sep + '{}_ROC_info_{}.npz'.format(prefix,col), tprs=tprs, mean_fpr=mean_fpr,
                        mean_tpr=mean_tpr, aucs=aucs)
    plt.plot(mean_fpr, mean_tpr,
             label=r'(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=1.5, alpha=.9)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_dir + os.sep + prefix + '_ROC_{}.png'.format(col), dpi=600, bbox_inches='tight')


"""**Train model for only EHR features**"""
def train_model_EHR(epoch, _batch, combine_data, label, model_dir,ROC_dir, prefix='', col=''):
    current_dir = model_dir + os.sep + col
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)

    _epoch = epoch
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits)
    i = 1
    predictions = []
    con_mat_avg = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    x = combine_data
    y = label

    x, y = shuffle(x, y, random_state=SEED)

    class_w = dict(zip(np.unique(label), class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(label), y=label)))

    for train_index, test_index in kf.split(x, y):
        y_train, y_test = y[train_index], y[test_index]

        ch1_train = np.stack(x[train_index, 1])
        ch1_test = np.stack(x[test_index, 1])

        model = get_model_EHR()

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            current_dir + os.sep + prefix + '_epoch_{}_{}.h5'.format(_epoch, i),
            monitor='val_accuracy', verbose=1,
            save_best_only=True, mode='max')
        log = tf.keras.callbacks.CSVLogger(current_dir + os.sep + prefix + '_epoch_{}_{}.log'.format(_epoch, i))

        print('\n')
        print('\n')
        print('**************************************************')
        print('training on fold {} out of {}'.format(i, n_splits))
        print('**************************************************')
        print('\n')
        print('\n')

        history = model.fit(
            ch1_train,
            y_train,
            # verbose=0,
            epochs=_epoch,
            batch_size=_batch,
            validation_data=(ch1_test,y_test),
            callbacks=[checkpoint,
                       log
                       ],
        )

        model.load_weights(current_dir + os.sep + prefix + '_epoch_{}_{}.h5'.format(_epoch, i))


        pred_score = model.predict(ch1_test)
        pred_ = (pred_score > 0.5)
        predictions.append(pred_)

        plt.figure()
        plt.plot(history.history['loss'], label='Binary crossentropy (training data)')
        plt.plot(history.history['val_loss'], label='Binart crossentropy (validation data)')
        plt.plot(history.history['accuracy'], label='Accuracy (training data)')
        plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
        plt.title('Model performance for 3D MNIST Keras Conv3D example')
        plt.ylabel('Loss value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.savefig(current_dir + os.sep + prefix + '_fold_{}.png'.format(i), dpi=600, bbox_inches='tight')

        con_mat = tf.math.confusion_matrix(labels=y_test, predictions=pred_).numpy()
        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        con_mat_avg.append(con_mat_norm)

        labels = ["0", "1"]

        con_mat_df = pd.DataFrame(con_mat_norm, index=labels, columns=labels)

        plt.figure()
        plt.rcParams.update({'font.size': 18})
        sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('confusion matrix fold {}'.format(i))
        plt.savefig(current_dir + os.sep + prefix + '_fold_{}.png'.format(i), dpi=600, bbox_inches='tight')

        fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(y_test, pred_score)
        tprs.append(np.interp(mean_fpr, fpr_keras, tpr_keras))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr_keras, tpr_keras)
        aucs.append(roc_auc)

        i += 1

    con_mat_avg = np.array(con_mat_avg)

    con_mat_df = pd.DataFrame(con_mat_avg.mean(axis=0), index=labels, columns=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix ICD-10 ({})'.format(col))
    plt.savefig(ROC_dir + os.sep + prefix + '_confusion_{}.png'.format(col), dpi=600,bbox_inches='tight')

    plt.figure(figsize=(10, 8))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    np.savez_compressed(ROC_dir + os.sep + '{}_ROC_info_{}.npz'.format(prefix,col), tprs=tprs, mean_fpr=mean_fpr,
                        mean_tpr=mean_tpr, aucs=aucs)
    plt.plot(mean_fpr, mean_tpr,
             label=r'(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=1.5, alpha=.9)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_dir + os.sep + prefix + '_ROC_{}.png'.format(col), dpi=600, bbox_inches='tight')



def training_fusion_model(latent_dic_path, labels_path, EHR_path):

    """**Load OPG latent space**"""
    latent_data = np.load(latent_dic_path)
    x = latent_data['Laten_data']

    """**Load labels**"""
    data_label = pd.read_csv(labels_path, index_col=0)

    """**Load Clinical**"""
    EHR_data = pd.read_csv(EHR_path, index_col=0)

    cols_to_norm = ['age','income', 'teeth_number', 'extent_bone_loss_>33%', 'Bone_loss_max', 'bone_loss_age']
    EHR_data[cols_to_norm] = EHR_data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    EHR_array = EHR_data.loc[:, :].values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(EHR_array)
    EHR_array = imp_mean.transform(EHR_array)
    np.argwhere(np.isnan(EHR_array))

    cols = ["I"]


    save_dir = './models/Final_v2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    datasets_dir = save_dir + os.sep + 'datasets'
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    for col in data_label.columns:
        prefix = 'latent'

        y = np.array(data_label[col])
        merge_db = np.concatenate((EHR_array,x), axis=1)

        over = SMOTENC(categorical_features=[1, 3, 4], random_state=SEED, sampling_strategy=0.5)
        under = RandomUnderSampler(random_state=SEED)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        try:
            X_sm, y_sm = pipeline.fit_resample(merge_db, y)
        except:
            X_sm = merge_db
            y_sm = y

        X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm,
                                                            stratify=y_sm,
                                                            shuffle=True,
                                                            random_state=SEED,
                                                            test_size=0.3)
        np.savez_compressed(datasets_dir + os.sep + '{}_dataset_{}.npz'.format(prefix, col), X_train=X_train, X_test=X_test,
                            y_train=y_train, y_test=y_test)

        EHR_X = X_train[:, :8]
        latent_X = X_train[:, 8:]

        combine_data = np.array(list(zip(latent_X, EHR_X)))
        encoder = LabelEncoder()
        encoder.fit(y_train)
        label = encoder.transform(y_train)

        model_dir = save_dir + os.sep + prefix
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        ROC_dir = save_dir + os.sep + 'ROC'
        if not os.path.exists(ROC_dir):
            os.makedirs(ROC_dir)

        train_model_OPG(200, 50, combine_data, label, model_dir, ROC_dir, prefix, col)


    for col in data_label.columns:
        prefix = 'EHR'
        y = np.array(data_label[col])

        merge_db = np.concatenate((EHR_array,x), axis=1)

        over = SMOTENC(categorical_features=[1, 3, 4], random_state=SEED, sampling_strategy=0.5)
        under = RandomUnderSampler(random_state=SEED)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        try:
            X_sm, y_sm = pipeline.fit_resample(merge_db, y)
        except:
            X_sm = merge_db
            y_sm = y
        X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm,
                                                            stratify=y_sm,
                                                            shuffle=True,
                                                            random_state=SEED,
                                                            test_size=0.3)
        np.savez_compressed(datasets_dir + os.sep + '{}_dataset_{}.npz'.format(prefix, col), X_train=X_train, X_test=X_test,
                            y_train=y_train, y_test=y_test)

        EHR_X = X_train[:, :8]
        latent_X = X_train[:, 8:]

        combine_data = np.array(list(zip(latent_X, EHR_X)))
        encoder = LabelEncoder()
        encoder.fit(y_train)
        label = encoder.transform(y_train)

        model_dir = save_dir + os.sep + prefix
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        ROC_dir = save_dir + os.sep + 'ROC'
        if not os.path.exists(ROC_dir):
            os.makedirs(ROC_dir)
        train_model_EHR(200, 50, combine_data, label, model_dir,ROC_dir, prefix, col)

    for col in data_label.columns:
        prefix = 'Both'
        y = np.array(data_label[col])

        merge_db = np.concatenate((EHR_array,x), axis=1)

        over = SMOTENC(categorical_features=[1, 3, 4], random_state=SEED, sampling_strategy=0.5)
        under = RandomUnderSampler(random_state=SEED)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        try:
            X_sm, y_sm = pipeline.fit_resample(merge_db, y)
        except:
            sm = SMOTE(random_state=SEED)
            X_sm, y_sm = sm.fit_resample(merge_db, y)
        X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm,
                                                            stratify=y_sm,
                                                            shuffle=True,
                                                            random_state=SEED,
                                                            test_size=0.3)
        np.savez_compressed(datasets_dir + os.sep + '{}_dataset_{}.npz'.format(prefix, col), X_train=X_train, X_test=X_test,
                            y_train=y_train, y_test=y_test)

        EHR_X = X_train[:, :8]
        latent_X = X_train[:, 8:]

        combine_data = np.array(list(zip(latent_X, EHR_X)))
        encoder = LabelEncoder()
        encoder.fit(y_train)
        label = encoder.transform(y_train)

        model_dir = save_dir + os.sep + prefix
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        ROC_dir = save_dir + os.sep + 'ROC'
        if not os.path.exists(ROC_dir):
            os.makedirs(ROC_dir)
        train_model_EHR_OPG(300, 50, combine_data, label, model_dir, ROC_dir, prefix, col)





