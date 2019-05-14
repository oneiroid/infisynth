import os
import numpy as np
import tensorflow as tf
from hparams import hparams
from models import create_model
from text import text_to_sequence
from util import audio
import time


class Synthesizer:
  def __init__(self):
    self.uid = 'parallel_{}'.format(int(time.time()))


  def load(self, checkpoint_path, model_name='tacotron'):
      print('Constructing model: %s' % model_name)
      inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
      input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
      with tf.variable_scope('model') as scope:
        self.model = create_model(model_name, hparams)
        self.model.initialize(inputs, input_lengths)
        self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

      print('Loading checkpoint: %s' % checkpoint_path)
      config = tf.ConfigProto(log_device_placement=False)
      config.intra_op_parallelism_threads = 1
      config.inter_op_parallelism_threads = 1

      #config.gpu_options.allow_growth = True

      self.session = tf.Session(config=config)
      self.session.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      saver.restore(self.session, checkpoint_path)

  def load_saved_model(self, path):
      config = tf.ConfigProto()
      #config.gpu_options.allow_growth = True
      self.session = tf.Session(config=config)
      self.session.run(tf.global_variables_initializer())
      tf.saved_model.loader.load(self.session, ['serve'], path)


  #def synthesize_saved_model(self, text, fn_wav='noname.wav'):



  def synthesize(self, text, fn_wav = 'noname.wav'):
      start= time.perf_counter()
      cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
      seq = text_to_sequence(text, cleaner_names)
      #print(seq)
      feed_dict = {
        self.model.inputs: [np.asarray(seq, dtype=np.int32)],
        self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
      }
      timepoint1 = time.perf_counter()
      print("inference prep time {:.2f}s/it".format(timepoint1 - start))

      start = time.time()
      wav = self.session.run(self.wav_output, feed_dict=feed_dict)
      timepoint1 = time.time()
      print("inference time {:.2f}s/it".format(timepoint1 - start))
      print(timepoint1 - start)

      start = time.perf_counter()
      wav = audio.inv_preemphasis(wav)
      wav = wav[:audio.find_endpoint(wav)]
      timepoint1 = time.perf_counter()
      print("inference post proc time {:.2f}s/it".format(timepoint1 - start))

      #audio.save_wav(wav, './' + fn_wav)
      return wav



synth = Synthesizer()
#synth.load('./models/tacotron-20180906/model.ckpt')
synth.load('./models/tacotron-20190510/model.ckpt-364000')


for i in range(5):
    te = 'Another world, another time. This land was green and good. Until the crystal cracked'
    synth.synthesize(te, str(int(time.time())) + '_1.wav')


    te = 'This is one very difficult sentense for me to pronounce, but I am trying my best, yo.'
    synth.synthesize(te, str(int(time.time())) + '_2.wav')

    te = 'Another world, another time. This land was green and good. Until the crystal cracked.'
    synth.synthesize(te, str(int(time.time())) + '_3.wav')

    te = 'This is one very difficult sentense for me to pronounce, but I am trying my best, yo'
    synth.synthesize(te, str(int(time.time())) + '_4.wav')

    te = 'Now lets see how much time will it take to synthesize something conplemetely different after the system has been warmed up.'
    synth.synthesize(te, str(int(time.time())) + '_5.wav')
