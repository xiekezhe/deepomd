
import tensorflow as tf
from tensorflow.python.platform import gfile

model = './models/leduc_poker_deep_cfr_value_0_cpu.pb'
new_saver = tf.train.import_meta_graph(model)
graph = tf.get_default_graph()
summaryWriter = tf.summary.FileWriter('log/', graph)
