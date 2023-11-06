import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from astronet.t2.model import T2Model_AG
from collections import Counter

# set gpus for schumann
gpus = tf.config.get_visible_devices('GPU')
tf.config.set_visible_devices(gpus[1], 'GPU')

print(f"Loading Data")
# Load data
X_train = np.load('X_train_ZTF_Sim_FullSliced_Padded_0pt2GP_hostPhotTrue_30Cut.npz')['arr_0']
X_test = np.load('X_test_ZTF_Sim_FullSliced_Padded_0pt2GP_hostPhotTrue_30Cut.npz')['arr_0']
y_train = np.load('y_train_ZTF_Sim_FullSliced_Padded_0pt2GP_hostPhotTrue_30Cut.npz')['arr_0']
y_test = np.load('y_test_ZTF_Sim_FullSliced_Padded_0pt2GP_hostPhotTrue_30Cut.npz')['arr_0']

print(Counter(y_train))

#binarize labels 
y_train = OneHotEncoder(max_categories=3, sparse_output=False).fit_transform(y_train.reshape(-1, 1))
y_test = OneHotEncoder(max_categories=3, sparse_output=False).fit_transform(y_test.reshape(-1, 1))

print(f"Setting up Parameters")
# Create Parameters for the model
params = {}
#weighting the classes - off for now
params['class_weight'] = {0:1, 1:1, 2:1}

#weighting in time to prioritize early classification
weights = np.ones(np.shape(X_train[:, :, 0]))
weights[(X_train[:, :, 0][:, -1] < 3)] = 10
weights[(X_train[:, :, 0][:, -1] > 3) & (X_train[:, :, 0][:, -1] < 15)] = 5

weights[y_train[:,0] == 0] *= params['class_weight'][0]
weights[y_train[:,0] == 1] *= params['class_weight'][1]
weights[y_train[:,0] == 2] *= params['class_weight'][2]

#compress -- not doing time-distributed network
weights = weights[:, 0]

# using the optimally-determined parameters here
params['num_classes'] = y_train.shape[1]
params['batch_size'] = 128
#params['epochs'] = 200
params['epochs'] = 10
params['filters']= 50
params['ff_dim'] = 32
params['embed_dim'] = 32
params['num_layers'] = 4
params['num_heads'] = 2
params['droprate'] = 0.28

# --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
params['num_filters'] = params['embed_dim']
params['passbands'] = 'XY'

print(f"Configuring save location")
outputfn = '0pt2GP_Transformer_wMask_FullObsTraining_OptParams'
outputdir = './'
modelPath = outputdir + '/models/'

(
    _,
    timesteps,
    num_features,
) = X_train.shape 
input_shape = (params['batch_size'], timesteps, num_features)
params['input_shape'] = input_shape

print(f"Creating model")
# Create the TF model object
model = T2Model_AG(
    input_dim=params['input_shape'],
    embed_dim=params['embed_dim'],
    num_heads=params['num_heads'],
    ff_dim=params['ff_dim'],
    num_filters=params['num_filters'],
    num_classes=params['num_classes'],
    num_layers=params['num_layers'],
    droprate=params['droprate'],
)

params['learning_rate'] = 5.e-4
opt = keras.optimizers.legacy.Adam(learning_rate=params['learning_rate'])
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

#model.build_graph(input_shapes=[params['input_shape']])
model(X_train[:32,:,:], training=True)

print(f"Loading model weights")
tf.train.Checkpoint(model=model) \
        .restore(modelPath+"/Model_%s_Checkpointweights.sav"%outputfn) \
        .expect_partial()
        #.assert_existing_objects_matched() \

print(f"Saving as saved model")
model.save("saved_model")
