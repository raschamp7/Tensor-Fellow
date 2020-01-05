""" 
A neural chatbot using seq2seq model with
attentional decoder. 

This file runs the chatbot.
It has 2 modes: trainTheBot and chatWithBot

It runs in python 2.7 and python 3.5

Based on:
Sequence to sequence model by Cho et al.(2014)
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
import sys
import time

import numpy as np
import tensorflow as tf

from model import ChatBotModel
import config
import data

def _getRandomBucket(trainBucketsScale):
    """ 
    Get a random bucket from which to choose a training sample 
    """
    rand = random.random()
    return min([i for i in range(len(trainBucketsScale))
                if trainBucketsScale[i] > rand])


def _getBuckets():
    """ 
    Load the dataset into buckets based on their lengths.
    train_buckets_scale is the inverval that'll help us 
    choose a random bucket later on.
    """
    test_buckets = data.loadData(config.TESTFILE+config.IDS+config.ENCODER, config.TESTFILE+config.IDS+config.DECODER)
    data_buckets = data.loadData(config.TRAINFILE+config.IDS+config.ENCODER, config.TRAINFILE+config.IDS+config.DECODER)
    train_bucket_sizes = [len(data_buckets[b]) for b in range(len(config.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    # list of increasing numbers from 0 to 1 that we'll use to select a bucket.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return test_buckets, data_buckets, train_buckets_scale

def _getSkipStep(iteration):
    """ 
    How many steps should the model trainTheBot before it saves all the weights. 
    """
    if iteration < config.CHECKPOINTSTEP:
        return config.CHECKPOINTSMALL
    return config.CHECKPOINTSTEP

def _checkRestoreParameters(sess, saver):
    """ 
    Restore the previously trained parameters if there are any. 
    """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/' + config.CHECKPT_FILE))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading checkpointed parameters for Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Found checkpointed data. Loading checkpoint....")
    else:
        print("Starting without a checkpoint. Fresh Chatbot!")

def _evalTestSet(sess, model, test_buckets):
    """ 
    Evaluate on the test set. 
    """
    for bucket_id in range(len(config.BUCKETS)):
        if len(test_buckets[bucket_id]) == 0:
            print("  Test: empty bucket %d" % (bucket_id))
            continue
        start = time.time()
        encoder_inputs, decoder_inputs, decoder_masks = data.getBatch(test_buckets[bucket_id], 
                                                                        bucket_id,
                                                                        batchSize=config.BATCH_SIZE)
        _, step_loss, _ = model.runStep(sess, encoder_inputs, decoder_inputs, 
                                   decoder_masks, bucket_id, True)
        print('Test bucket[{}] =( loss {:3.3f}: time {:3.3f} s )'.format(bucket_id, step_loss, time.time() - start))

def trainTheBot():
    """ 
    Training function
    """
    print("This is a training session....")
    test_buckets, data_buckets, train_buckets_scale = _getBuckets()
    # in trainTheBot mode, we need to create the backward path, so forwrad_only is False
    model = ChatBotModel(False, config.BATCH_SIZE, useLstm=config.globalUseLstm ,useAdam=config.globalUseAdam)
    model.buildGraph()

    saver = tf.train.Saver(max_to_keep=config.MAX_TO_KEEP)

    with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True)) as sess:
        print('Start training session')
        sess.run(tf.global_variables_initializer())
        _checkRestoreParameters(sess, saver)

        iteration = model.globalStep.eval()
        total_loss = 0
        #smallTotal = 0
        start = time.time()
        while True:
            try:
                skip_step = _getSkipStep(iteration)
                bucket_id = _getRandomBucket(train_buckets_scale)
                encoder_inputs, decoder_inputs, decoder_masks = data.getBatch(data_buckets[bucket_id], 
                                                                               bucket_id,
                                                                               batchSize=config.BATCH_SIZE)               
                _, step_loss, _ = model.runStep(sess, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
                total_loss += step_loss
                # smallTotal += step_loss
                iteration += 1
    
#                 if iteration % config.CHECKPOINTSMALL == 0:
#                     tf.summary.scalar('StepLoss', tf.float32(smallTotal/config.CHECKPOINTSMALL))
#                     smallTotal = 0
    
                if iteration % skip_step == 0:
                    print('Epoch [{:3.3f}] = ( Average Step Loss {:3.3f}: Average Step Time {:3.3f} s )'.format(iteration/1000, total_loss/skip_step, (time.time() - start)/skip_step))
                    start = time.time()
                    total_loss = 0
                    saver.save(sess, os.path.join(config.CPT_PATH, 'chatbot'), global_step=model.globalStep)
                    if iteration % (config.CHECKTESTMULT * skip_step) == 0:
                        # Run evals on development set and print their loss
                        _evalTestSet(sess, model, test_buckets)
                        start = time.time()
                    sys.stdout.flush()
            except KeyboardInterrupt:
                print("Keyboard interrupt received... Exiting")
                return

def _getUserInput():
    """ 
    Get user's input, which will be transformed into encoder input later 
    """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()

def _findRightBucket(length):
    """ 
    Find the proper bucket for an encoder input based on its length 
    """
    return min([b for b in range(len(config.BUCKETS))
                if config.BUCKETS[b][0] >= length])
    
def _isQuestion(word):
    question = {'how': True, 'hows': True, 'where': True,'who': True, 'whos': True, 'why': True, 
                'what': True, 'are': True, 'do' : True, 'whats': True}
    try:
        return question[word]
    except KeyError:
        return False
 
    
def _dictWithPunctuation(encVocab):
    try:
        _ = encVocab['!']
        return True
    except KeyError:
        return False

def _dictWithApostroph(encVocab):
    try:
        _ = encVocab['\'']
        return True
    except KeyError:
        return False
    
    
def _constructResponse(output_logits, inv_dec_vocab, enc_vocab):
    """ 
    Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB
    
    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """
    #print(output_logits[0])

    outputs = []
    """
    we use a factor to multiply the best answer other than 
    EOS. This will make the response more chatty
    The factor is reduced to factor**0.3
    every iteration. See below.
    """
    if config.useFactor == True:
        factor = config.globalFactor
    else:
        factor = 1
    for logit in output_logits:
        nonstop = np.argmax(logit[0][config.END_ID+1:])+config.END_ID+1
        #if iter == 0:
        #   outputs.append(int(nonstop))
        #else:
        if config.globalPrintDebug == True:
            print("Factor={} EOS={} Token={}".format(factor, logit[0][config.END_ID], logit[0][nonstop]))
        if logit[0][config.END_ID] > factor* logit[0][nonstop]:
            outputs.append(config.END_ID)
            factor = factor**config.globalDecay
        else:
            # print("EOS={} Token={}".format(logit[0][config.END_ID], logit[0][nonstop]))
            factor = factor**config.globalDecay
            try:
                if config.useFactor == True:
                    if outputs[-1] == config.END_ID:
                        if _dictWithPunctuation(enc_vocab):
                            outputs[-1] = enc_vocab['.']
                        else:
                            outputs[-1] = config.START_ID
                        if config.useFactor == True:
                            factor = config.globalFactor
                        else:
                            factor = 1
            except IndexError:
                pass
            outputs.append(int(nonstop))
            #if logit[0][config.END_ID] > logit[0][nonstop]:

    #outputs = [int(np.argmax(logit[0][config.END_ID:])+config.END_ID) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if config.globalPrintDebug == True:
        print("Outputs{}".format(outputs))
        for output in outputs:
            print("OToken={}".format(inv_dec_vocab[output]))
    if config.END_ID in outputs:
        outputs = outputs[:outputs.index(config.END_ID)]
    # Print out sentence corresponding to outputs.
    if _dictWithPunctuation(enc_vocab):
        eol = ''
    else:
        try:
            first = outputs[0]
            if _isQuestion(inv_dec_vocab[first]):
                eol = " ?"
            else:
                eol = " ."
        except IndexError:
            eol = " ???"
    
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs]) + eol

def chatWithBot():
    """ 
    this is the main chat function
    it rebuilds the model without backprobagation and waits for the user to input
    a line. it runs the line through the model the outputs a result
    """
    inv_dec_vocab , enc_vocab = data.loadVocabulary(os.path.join(config.PROCESSED_PATH, config.VOCAB_FILE))
    
    if not _dictWithPunctuation(enc_vocab):
        inv_dec_vocab[config.START_ID] = '.'
    
    
    model = ChatBotModel(True, batchSize=1, useLstm= config.globalUseLstm, useAdam=config.globalUseAdam)
    model.buildGraph()

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(config.CPT_PATH, sess.graph)
        sess.run(tf.global_variables_initializer())
        _checkRestoreParameters(sess, saver)
        output_file = open(config.OUTPUT_FILE, 'a+')
        # Decode from standard input.
        max_length = config.BUCKETS[-1][0]
        print('Welcome to {}! Press <enter> at prompt to exit!'.format(config.PROGNAME))
        if _dictWithPunctuation(enc_vocab):
            config.USEPUNCTUATION = True
            if _dictWithApostroph(enc_vocab):
                config.USEAPO = True
            else:
                config.USEAPO = False
            print('This instance uses punctuation. Use [{}] to improve the answers.'.format(config.PUNCTCHAR))
        else:
            config.USEPUNCTUATION = False
            config.USEAPO = False
            print('This instance ignores punctuation. Any typed punctuation will be stripped from user input.')
        
        while True:
            line = _getUserInput()
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]
            if line == '':
                break
            output_file.write('HUMAN ++++ ' + line + '\n')
            # Get token-ids for the input sentence.
            token_ids = data.sentence2ID(enc_vocab, str(line))
            if (len(token_ids) > max_length):
                print('Max length I can handle is:', max_length)
                continue
            # Which bucket does it belong to?
            bucket_id = _findRightBucket(len(token_ids))
            if config.globalPrintDebug == True:
                print("BucketID {} Token Ids {}".format(bucket_id, token_ids))
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_masks = data.getBatch([(token_ids, [])], 
                                                                            bucket_id,
                                                                            batchSize=1)
            # Get output logits for the sentence.
            _, _, output_logits = model.runStep(sess, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            response = _constructResponse(output_logits, inv_dec_vocab, enc_vocab)
            print('\n'+'BOT ++++ ' + response + '\n')
            output_file.write('BOT ++++ ' + response + '\n')
        output_file.write('=========={}==LAYERS={}==HIDENSIZE={}==BATCH={}==BUCKETS={}==PUNCTUATION={}===USEFACTOR={}=====\n'.format(time.ctime(), config.NUM_LAYERS, config.HIDDEN_SIZE, config.BATCH_SIZE, config.BUCKETS, config.USEPUNCTUATION, config.useFactor))
        output_file.close()
        writer.close() 

def testTheBot():
    """ 
    this function runs lists of user inputs from a file
    it rebuilds the model without backprobagation and waits for the user to input
    a line. it runs the line through the model the outputs a result
    it is used to test model with specific batches of inputs
    as all results are recorded in output_convo.txt we can check which 
    model is 'better'
    """    
    inv_dec_vocab , enc_vocab = data.loadVocabulary(os.path.join(config.PROCESSED_PATH, config.VOCAB_FILE))
    
    if not _dictWithPunctuation(enc_vocab):
        inv_dec_vocab[config.START_ID] = '.'
    
    model = ChatBotModel(True, batchSize=1, useLstm= config.globalUseLstm, useAdam=config.globalUseAdam)
    model.buildGraph()

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(config.CPT_PATH, sess.graph)
        sess.run(tf.global_variables_initializer())
        _checkRestoreParameters(sess, saver)
        output_file = open(config.OUTPUT_FILE, 'a+')
        # Decode from standard input.
        max_length = config.BUCKETS[-1][0]
        if _dictWithPunctuation(enc_vocab):
            config.USEPUNCTUATION = True
            if _dictWithApostroph(enc_vocab):
                config.USEAPO = True
            else:
                config.USEAPO = False
            print('This instance uses punctuation. Use [{}] to improve the answers.'.format(config.PUNCTCHAR))
        else:
            config.USEPUNCTUATION = False
            config.USEAPO = False
            print('This instance ignores punctuation. Any typed punctuation will be stripped from user input.')

        try:
            with open(config.CMDFILENAME, 'r') as file:
                for line in file.readlines():
                    if len(line) > 0 and line[-1] == '\n':
                        line = line[:-1]
                    if line == '':
                        break
                    output_file.write('HUMAN ++++ ' + line + '\n')
                    # Get token-ids for the input sentence.
                    token_ids = data.sentence2ID(enc_vocab, str(line))
                    if (len(token_ids) > max_length):
                        print('Max length I can handle is:', max_length)
                        continue
                    # Which bucket does it belong to?
                    bucket_id = _findRightBucket(len(token_ids))
                    if config.globalPrintDebug == True:
                        print("BucketID {} Token Ids {}".format(bucket_id, token_ids))
                    # Get a 1-element batch to feed the sentence to the model.
                    encoder_inputs, decoder_inputs, decoder_masks = data.getBatch([(token_ids, [])], 
                                                                                    bucket_id,
                                                                                    batchSize=1)
                    # Get output logits for the sentence.
                    _, _, output_logits = model.runStep(sess, encoder_inputs, decoder_inputs,
                                                   decoder_masks, bucket_id, True)
                    response = _constructResponse(output_logits, inv_dec_vocab, enc_vocab)
                    print('\n'+'BOT ++++ ' + response + '\n')
                    output_file.write('BOT ++++ ' + response + '\n')
                file.close()
        except FileNotFoundError:
            print('error: Test file {} not found!'.format(config.CMDFILENAME))

        output_file.write('=========={}==LAYERS={}==HIDENSIZE={}==BATCH={}==BUCKETS={}==PUNCTUATION={}===USEFACTOR={}=====\n'.format(time.ctime(), config.NUM_LAYERS, config.HIDDEN_SIZE, config.BATCH_SIZE, config.BUCKETS, config.USEPUNCTUATION, config.useFactor))
        output_file.close()
        writer.close() 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat', 'test'},
                        default='chat', help="mode. train will train the bot, chat will allow user to write line, test will test against a fixed set of entrie")
    parser.add_argument('--debug', choices = {'yes','no'},
                        default='no', help="debug. prints extra information on the command line when chatting")
    parser.add_argument('--opt', choices = {'greedy','adam'},
                        default='adam', help="opt. set the optimization algorithm. default is greedy sdg")
    parser.add_argument('--cell', choices = {'gru','lstm'},
                        default='gru', help="cell. set the rnn cell type. default is gru")
    parser.add_argument('--usefactor', choices = {'yes','no'},
                        default='no', help="use response weight factor to build the answer")
    parser.add_argument('--setfactor', type=float,
                        default=1.5, help="set the answer verbosity factor")
    parser.add_argument('--maxtokeep', type=int,
                        default=20, help="how many checkpoints should we keep on disk")
    parser.add_argument('--testfile', type=str,
                         default='test.txt', help="use this as test file. test mode only")
    
    args = parser.parse_args()
    
    if not os.path.isdir(config.PROCESSED_PATH):
        data.prepareRawData()
        data.processData()
    print('Data ready!')
    # create checkpoints folder if there isn't one already
    data.makeOutputDirectory(config.CPT_PATH)

    if args.usefactor == 'yes':
        config.useFactor = True
        print("Use factor On")
    else:
        print("Use factor Off")
        config.useFactor = False
            
    if args.debug == 'yes':
        config.globalPrintDebug = True
        print("Debug Mode On")
    else:
        print("Debug Mode Off")
        config.globalPrintDebug = False
        
    if args.cell == 'lstm':
        print("Using LSTM Cells")
        config.globalUseLstm = True
    else:
        print("Using GRU Cells")
        config.globalUseLstm = False

    if args.opt == 'adam':
        print("Using Adam Optimizer")
        config.globalUseAdam = True
    else:
        print("Using Greedy Optimizer")
        config.globalUseAdam = False
        
    config.globalFactor = args.setfactor
    config.MAX_TO_KEEP = args.maxtokeep
    config.CMDFILENAME = args.testfile
    
    if args.mode == 'train':
        trainTheBot()
    elif args.mode == 'chat':
        chatWithBot()
    elif args.mode == 'test':
        testTheBot()

if __name__ == '__main__':
    main()
