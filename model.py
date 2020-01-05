""" 
Bot Model using seq2seq neural net 
ICS4U Winter 2017

Uses the seq2seq model that was originally
invented for language translation
The chat model "translates" from english
to english.

Works with python2.7 and python3.5

Based on:
Sequence to sequence model by Cho et al.(2014)
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

"""

import time
import tensorflow as tf
import numpy as np

import config
import os
#from tensorflow.contrib.slim.python.slim import learning

class ChatBotModel(object):
    """
    Sequence-to-sequence model with attention decoder and multiple buckets.
      This class implements a multi-layer recurrent neural network as encoder,
      and an attention-based decoder. This is the same as the model described in
      this paper: http://arxiv.org/abs/1412.7449.
      The model was invented for language to language translation
      and is implemented by google and is part of
      tensorflow. We need to configure the model with appropriate parameters
      to see if it can be used for a chatbot.
      This class also allows to use GRU cells in addition to LSTM cells, and
      sampled softmax to handle large output vocabulary size. 
      Softmax, or normalized exponential function is decribed here:
       https://en.wikipedia.org/wiki/Softmax_function
      
      Data Members:
      vocabSize: size of the vocabulary. The seq2seq class uses 2 vocabularies
      one for the source language and one for the target language. For
      chatting we need only one vocabulary that is passed to both sourceVocab
      and targetVocab in the seq2seq model
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
        These are specified in config.py and used from there
        Is is very important that the program is started in chat mode with
        the same buckets as it was using when training. To achieve that
        we save the config.py file in the checkpoint directory.
      hiddenSize: number of hidden states in each cell. It is configured 
        via config.HIDDEN_SIZ in config.py
      numLayers: number of layers in the model. This is the number of cells that
        are stacked (model depth). It is configured via config.NUM_LAYERS in config.py
      maxGradientNorm: gradients will be clipped to maximally this norm. Is is configured via
        config.MAX_GRAD_NORM
      batchSize: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding. Read from
        config.py
      learningRate: learning rate to start with.
      learningRateDecayFactor: decay learning rate by this much when needed.
      useLstm: if true, we use LSTM cells instead of GRU cells.
      numSamples: number of samples from the vocabulary used to build the sampled softmax.
      forwardNetworkOnly: if set, we do not construct the backward pass in the model.
    """
    def __init__(self, forwardOnly = False, batchSize = 1, useLstm = False, useAdam = False):
        """
        parameters: @forwardOnly: if true - do no construct backpropagation, else do
                    @batchSize - the list of batches
        """
        print('Create Seq2seq model with buckets...')
        self.forwardNetworkOnly = forwardOnly
        self.batchSize = batchSize
        self.useLstm = useLstm
        self.useAdam = useAdam
        setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
        setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
        setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)
        try:
            self.vocabSize = sum(1 for _ in open(os.path.join(config.PROCESSED_PATH, config.VOCAB_FILE)))
        except OSError:
            print("error: Vocabulary not found!")
    
    def _createPlaceholders(self):
        # Feeds for inputs. It's a list of placeholders
        print('Create TensorFlow placeholders for enc, dec and mask objects')
        self.encoderInputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][0])]
        self.decoderInputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][1] + 1)]
        self.decoderMasks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(config.BUCKETS[-1][1] + 1)]

        # Our targets are decoder inputs shifted by one (to ignore <s> symbol)
        self.targets = self.decoderInputs[1:]
 
         
    def _createCells(self):
        """
        Create the multi rnn cell using either GRU or LSTM cells
        Each cell maintains a HIDDEN_SIZE state and there
        are NUM_LAYERS cells in the model
        """
        print('Create MultiRNN cell with. layers = {}'.format(config.NUM_LAYERS))
        if self.useLstm == False:      
            print('Using GRU Cells to build the model. Hidden Size={}'.format(config.HIDDEN_SIZE))
            single_cell = tf.nn.rnn_cell.GRUCell(config.HIDDEN_SIZE)
        else:
            print('Using BasicLSTM Cells to build the model. Hidden Size={}'.format(config.HIDDEN_SIZE))
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(config.HIDDEN_SIZE)

        self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * config.NUM_LAYERS)

    def _createSeq2SeqModel(self):
        print('Create the Seq2S2q Model ...')
        print('Create a sampled softmax function')
        # If we use sampled softmax, we need an output projection.
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < self.vocabSize:
            w = tf.get_variable('proj_w', [config.HIDDEN_SIZE, self.vocabSize])
            b = tf.get_variable('proj_b', [self.vocabSize])
            self.outputProjection = (w, b)

        def sampledLoss(labels=None, logits=None):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(tf.transpose(w), b, labels, logits, 
                                              config.NUM_SAMPLES, self.vocabSize)
        self.softmaxLossFunction = sampledLoss

        print('Create the model...')
        start = time.time()
        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, self.cell,
                    num_encoder_symbols=self.vocabSize,
                    num_decoder_symbols=self.vocabSize,
                    embedding_size=config.HIDDEN_SIZE,
                    output_projection=self.outputProjection,
                    feed_previous=do_decode)

        if self.forwardNetworkOnly:
            print('Forward network model')
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoderInputs, 
                                        self.decoderInputs, 
                                        self.targets,
                                        self.decoderMasks, 
                                        config.BUCKETS, 
                                        lambda x, y: _seq2seq_f(x, y, True),
                                        softmax_loss_function=self.softmaxLossFunction)
            # If we use output projection, we need to project outputs for decoding.
            if self.outputProjection:
                for bucket in range(len(config.BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output, 
                                            self.outputProjection[0]) + self.outputProjection[1]
                                            for output in self.outputs[bucket]]
        else:
            print('Network with back-propagation')
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoderInputs, 
                                        self.decoderInputs, 
                                        self.targets,
                                        self.decoderMasks,
                                        config.BUCKETS,
                                        lambda x, y: _seq2seq_f(x, y, False),
                                        softmax_loss_function=self.softmaxLossFunction)
        print('Time: {:3.3f} seconds'.format(time.time() - start))

    def _createOptimizer(self):
        """
        The optimizer is set only in training mode as it is used to find the model weights
        """

        with tf.variable_scope('training') as _:
            print('Creating optimizer function (one per bucket)...')
            self.globalStep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.forwardNetworkOnly:
                if self.useAdam == True:
                    self.optimizer = tf.train.AdamOptimizer()
                else:
                    self.optimizer = tf.train.GradientDescentOptimizer(config.LR)
                trainables = tf.trainable_variables()
                self.gradientNorms = []
                self.trainOps = []
                start = time.time()
                for bucket in range(len(config.BUCKETS)):
                    
                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket], 
                                                                 trainables),
                                                                 config.MAX_GRAD_NORM)
                    self.gradientNorms.append(norm)
                    self.trainOps.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables), 
                                                            global_step=self.globalStep))
                    print('Created optimized function for bucket {} in {:3.3f} seconds'.format(bucket, time.time() - start))
                    start = time.time()


    def _createSummary(self):
        pass

        
    def _assertLengths(self, encoderSize, decoderSize, encoderInputs, decoderInputs, decoderMasks):
        """
         Assert that the encoder inputs, decoder inputs, and decoder masks are
         of the expected lengths 
        """
        if len(encoderInputs) != encoderSize:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                            " %d != %d." % (len(encoderInputs), encoderSize))
        if len(decoderInputs) != decoderSize:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(decoderInputs), decoderSize))
        if len(decoderMasks) != decoderSize:
            raise ValueError("Weights length must be equal to the one in bucket,"
                           " %d != %d." % (len(decoderMasks), decoderSize))


    def buildGraph(self):
        self._createPlaceholders()
        self._createCells()
        self._createSeq2SeqModel()
        self._createOptimizer()
        self._createSummary()

        
    def runStep(self, sess, encoderInputs, decoderInputs, decoderMasks, bucketId, forwardOnly):
        """ 
        Run one step in training.
         Args:
          session: tensorflow session to use.
          encoderInputs: list of numpy int vectors to feed as encoder inputs.
          decoderInputs: list of numpy int vectors to feed as decoder inputs.
          decoderMasks: list of numpy float vectors to feed as target weights.
          bucketId: which bucket of the model to use.
          forwardOnly: whether to do the backward step or only forward.
          forwardOnly is set to True when you just want to evaluate on the test set,
        or when you want to the bot to be in chatWithBot mode. 
        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        Raises (in _assertLengths):
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        encoder_size, decoder_size = config.BUCKETS[bucketId]
        self._assertLengths(encoder_size, decoder_size, encoderInputs, decoderInputs, decoderMasks)
    
        # input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for step in range(encoder_size):
            input_feed[self.encoderInputs[step].name] = encoderInputs[step]
        for step in range(decoder_size):
            input_feed[self.decoderInputs[step].name] = decoderInputs[step]
            input_feed[self.decoderMasks[step].name] = decoderMasks[step]
    
        last_target = self.decoderInputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batchSize], dtype=np.int32)
    
        # output feed: depends on whether we do a backward step or not.
        if not forwardOnly:
            output_feed = [self.trainOps[bucketId],  # update op that does SGD.
                           self.gradientNorms[bucketId],  # gradient norm.
                           self.losses[bucketId]]  # loss for this batch.
        else:
            output_feed = [self.losses[bucketId]]  # loss for this batch.
            for step in range(decoder_size):  # output logits.
                output_feed.append(self.outputs[bucketId][step])
    
        outputs = sess.run(output_feed, input_feed)
        if not forwardOnly:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.


