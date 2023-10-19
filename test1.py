import tensorflow as tf
from tensorflow import keras
import losses, train, models

basic_model = models.buildin_models("r18", dropout=0.4, emb_shape=512, output_layer="E")
# basic_model = models.buildin_models("MobileNet", dropout=0, emb_shape=256, output_layer="GDC")
data_path = './datasets/custom_BITS_aligned_112_112/'
# eval_paths = ['./datasets/faces_emore/lfw.bin', './datasets/faces_emore/cfp_fp.bin', './datasets/faces_emore/agedb_30.bin']
eval_paths = []

tt = train.Train(data_path, save_path='/resnet18_custom_BITS_ep50_bs16_emore.h5', eval_paths=eval_paths,
                basic_model=basic_model, batch_size=1, random_status=0,
                lr_base=0.1, lr_decay=0.5, lr_decay_steps=16, lr_min=1e-5)
# optimizer = tfa.optimizers.SGDW(learning_rate=0.1, momentum=0.9, weight_decay=5e-5)
optimizer = tf.optimizers.SGD(learning_rate=0.1, momentum=0.9   )
# optimizer = tf.optimizers.Adam(learning_rate=0.1, momentum=0.9   )
sch = [
  # {"loss": losses.ArcfaceLoss(scale=16), "epoch": 5, "optimizer": optimizer},
  # {"loss": losses.ArcfaceLoss(scale=32), "epoch": 5},
  {"loss": losses.AdaFaceLoss(scale=64,), "epoch": 50},
  # {"loss": losses.AdaFaceLoss(scale=64,), "epoch": 5},
  # {"loss": losses.ArcfaceLoss(scale=64), "epoch": 40},
  # {"loss": losses.AdaFaceLoss() , "epoch": 40, "optimizer": keras.optimizers.Adam()},
  # {"loss": losses.ArcfaceLoss(), "epoch": 20, "triplet": 64, "alpha": 0.35},
]
tt.train(sch, 0)


