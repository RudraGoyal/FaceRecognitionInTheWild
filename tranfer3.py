from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import keras
from keras.applications import ResNet101V2
from keras import layers
from keras.layers import Activation,Dense
from keras import models
from keras.models import load_model
from keras.applications import ResNet101V2
from keras.applications.resnet_v2 import preprocess_input

from models import NormDense
import train,losses

model_path = './checkpoints/adaface_ir101_webface12m_rgb.h5'
pretrained_model = load_model(model_path, compile=True)

# pretrained_model = ResNet101V2(weights='imagenet' , include_top=False , input_shape=(224,224,3))

for layer in pretrained_model.layers[:-10]:
    layer.trainable = False

# model = models.Sequential()

num_classes = 3
# model.add(pretrained_model)
# model.layers.pop()
# model.add(layers.Dense(256))
# model.add(layers.Dense(128,activation='relu'))
# model.add(layers.Dense(num_classes,activation='softmax'))

# pretrained_model.layers.pop()
# pretrained_model.add(layers.Dense(256))
# pretrained_model.add(layers.Dense(128,activation='relu'))
# pretrained_model.add(layers.Dense(num_classes,activation='softmax'))

# model.compile(loss=losses.AdaFaceLoss)


# keras.models.save_model(model,'./checkpoints/midway_model.h5',overwrite=True,save_format='h5')
# model.summary()
# pretrained_model.summary()

data_path = './datasets/custom_BITS_aligned_112_112/'
# eval_paths = ['./datasets/faces_emore/lfw.bin', './datasets/faces_emore/cfp_fp.bin', './datasets/faces_emore/agedb_30.bin']
eval_paths = []

tt = train.Train(data_path, save_path='adaface_ir101_webface12m_rgb_customdata.h5', eval_paths=eval_paths,
                basic_model=pretrained_model, batch_size=4, random_status=0,
                lr_base=0.1, lr_decay=0.5, lr_decay_steps=16, lr_min=1e-5)

sch = [
  {"loss": losses.AdaFaceLoss(scale=64,), "epoch": 50},
]
tt.train(sch, 0)