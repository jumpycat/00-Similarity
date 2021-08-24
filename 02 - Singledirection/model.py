import warnings
import os

def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from data_process import *

dst_train_path = r'D:\DATA\DeepFake\train/'
dst_test_path = r'D:\DATA\DeepFake\train/'

def model():
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

model = model()
model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])
model.fit_generator(generator=train_gen(), steps_per_epoch=100, epochs=20, verbose=1)
