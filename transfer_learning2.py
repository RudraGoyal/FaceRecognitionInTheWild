from keras.applications import ResNet101V2
import tensorflow as tf
from keras.layers import Activation,Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import keras


import losses, train

imag_rows, img_cols = 112,112

data_path = './datasets/custom_BITS_aligned_112_112'
eval_paths = []
num_classes = 3
batch_size = 8

# pretrained_model = keras.models.load_model('./checkpoints/midway_model.h5',compile=True)

# pretrained_model.summary()

# x = pretrained_model.output
# predictions = Dense(num_classes,activation='softmax')(x)
# model = Model(inputs=pretrained_model.input,outputs = predictions)

# # Specify the path to your dataset folder
dataset_path = "./datasets/custom_BITS_aligned_112_112"

# # model.compile(optimizer='adam',loss=losses.AdaFaceLoss(scale=64),metrics=['accuracy'])

# # model.fit(train_generator,epochs=10,verbose=1,initial_epoch=0,
# #             use_multiprocessing=True,
# #             workers=4)

# # pretrained_model = Activation(tf.nn.softmax())(pretrained_model)
# # pretrained_model.add()

tt = train.Train(data_path, './transfer_custombits.h5', eval_paths, model='./checkpoints/r18_s64_ep100_bs1_custom2.h5',
                batch_size=512, random_status=0, lr_base=0.1, lr_decay=0.5, lr_decay_steps=16, lr_min=1e-5,)

sch = [
  {"loss": losses.AdaFaceLoss(scale=64,), "epoch": 50},
#   {"loss": losses.AdaFaceLoss(scale=64,), "epoch": 5},
#   {"loss": losses.AdaFaceLoss() , "epoch": 40, "optimizer": keras.optimizers.Adam()},
]
tt.train(sch, 0)