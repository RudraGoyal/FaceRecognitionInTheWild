from keras.applications import ResNet101V2
import tensorflow as tf
from keras.layers import Activation,Dense
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import keras

import losses, train

imag_rows, img_cols = 112,112

data_path = './datasets/custom_BITS_aligned_112_112'
eval_paths = []
num_classes = 3
batch_size = 8

pretrained_model = keras.models.load_model('./checkpoints/resnet101v2_custom ep50_emore.h5',compile=True)

for layer in pretrained_model.layers[:-4]:
    layer.trainable = False

# pretrained_model.layers.pop()
print([pretrained_model.layers])

# pretrained_model.compile(loss=losses.AdaFaceLoss, optimizer='adam', metrics=['accuracy'])
# pretrained_model.layers.add(Dense(num_classes,activation='softmax'))

pretrained_model.save('./checkpoints/midway_model.h5',overwrite=True,)
pretrained_model.summary()

# x = pretrained_model.output
# predictions = Dense(num_classes,activation='softmax')(x)
# model = Model(inputs=pretrained_model.input,outputs = predictions)

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2  # Split the dataset into training and validation sets
# )

# # Specify the path to your dataset folder
# dataset_path = "./datasets/custom_BITS"

# # Set up the train and validation generators
# train_generator = train_datagen.flow_from_directory(
#     dataset_path,
#     target_size=(112, 112),  # Specify the input size expected by the model
#     batch_size=1,
#     class_mode='categorical',
#     subset='training'  # Use the training subset for training
# )

# model.compile(optimizer='adam',loss=losses.AdaFaceLoss(scale=64),metrics=['accuracy'])

# model.fit(train_generator,epochs=10,verbose=1,initial_epoch=0,
#             use_multiprocessing=True,
#             workers=4)

# pretrained_model = Activation(tf.nn.softmax())(pretrained_model)
# pretrained_model.add()

# tt = train.Train(data_path, './transfer_custombits.h5', eval_paths, model=pretrained_model,
#                 batch_size=512, random_status=0, lr_base=0.1, lr_decay=0.5, lr_decay_steps=16, lr_min=1e-5)

# sch = [
#   {"loss": losses.AdaFaceLoss(scale=64,), "epoch": 50},
  # {"loss": losses.AdaFaceLoss(scale=64,), "epoch": 5},
  # {"loss": losses.AdaFaceLoss() , "epoch": 40, "optimizer": keras.optimizers.Adam()},
# ]
# tt.train(sch, 0)