""" 
Cornell Movies Dialog data corpus processing file
This file creates all the subfiles for
seq2seq model training and chat sessions

ICS4U Project Winter 2017
Works for
python 2.7 and 3.5

Based on:
Sequence to sequence model by Cho et al.(2014)
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

"""
from __future__ import division
from __future__ import print_function
import numpy as np
#import matplotlib.pyplot as plt
import os
import random
import re

import config

def getLines():
    """ 
    Get script lines from the lines file 
    This is a textual representation 
    of the lines spoken in the movies
    """
    id2Line = {}
    filePath = os.path.join(config.DATA_PATH, config.LINE_FILE)
    with open(filePath, 'r') as file:
        allLines = file.readlines()
        for line in allLines:
            """
            split the line using the ' +++$+++ ' separator
            the line will split into 4 token
            Example line:
            +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Forget French.
            ['u0', 'm0', 'BIANCA', 'Forget French.']
            """
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                if parts[4][-1] == '\n':
                    parts[4] = parts[4][:-1]
                id2Line[parts[0]] = parts[4]
    return id2Line

def getConversations():
    """ 
    Get conversations from the movies dialog file 
    Conversations are lists of movie lines
    ie. [L1, L2, L3, L4]
    we pick pairs from the conversation list
    """
    filePath = os.path.join(config.DATA_PATH, config.CONVO_FILE)
    conversationList = []
    with open(filePath, 'r') as file:
        for line in file.readlines():
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                aConversation = []
                for line in parts[3][1:-2].split(', '):
                    aConversation.append(line[1:-1])
                #print(aConversation)
                conversationList.append(aConversation)

    return conversationList

def questionsAndAnswers(linesList, conversationsList):
    """ 
    Divide  the conversations in 2 sets:
    One set is used in the encoder and represents
    the "questions" and the other is used by
    the decodes and represents the "answers"
    Each file is produced by extracting
    lines from the lines file indexed by
    values obtained from the conversations
    file.
    """
    questions = []
    answers   = [] 
    for conversation in conversationsList:
        for index, _ in enumerate(conversation[:-1]):
            questions.append(linesList[conversation[index]])
            answers.append(linesList[conversation[index + 1]])
    """
    check if the length of the 2 structures is the same
    we want 1q / 1a as we are trying to train a chatbot
    if lengths are different the assert statement will
    stop the program here.
    For each line in the questions list there is a
    corresponding line in the answers list
    """
    assert len(questions) == len(answers)
    return questions, answers

def prepareDataset(questions, answers):
    """
    create a directory to store all the train & test encoder & decoder
    the directory name is user settable in the config file
    """
    makeOutputDirectory(config.PROCESSED_PATH)
    
    """
    Split the data set into train and test
    For test:
        Build 2 test filenamesWithPathList (test.enc and test.dec)
        Pick a set of line ids at random
        Use the ids to build a test file
        one for encoding and one decoding
    For train:
        Use ids not picked for test and write
        2 filenamesWithPathList (train.enc and train.dec)
    Both the file names and the terminations
    are configurable in config.py
    """
    testIds = random.sample([i for i in range(len(questions))],config.TESTSET_SIZE)
    
    filenamesList = [config.TRAINFILE+ config.ENCODER, config.TRAINFILE+config.DECODER, config.TESTFILE+config.ENCODER, config.TESTFILE+config.DECODER]
    
    filenamesWithPathList = []
    for filename in filenamesList:
        filenamesWithPathList.append(open(os.path.join(config.PROCESSED_PATH, filename),'w'))

    dataBuckets = config.BUCKETS;
    maxInput = dataBuckets[len(dataBuckets)-1][0]
    maxOutput = dataBuckets[len(dataBuckets)-1][1]
    for i in range(len(questions)):
        """
        keep only dialog that is within QADIFF_THRESHOLD words of each other in length
        
        """
        qTokens = lineTokenizer(questions[i])
        aTokens = lineTokenizer(answers[i])
        ql = len(qTokens)
        al = len(aTokens)
        m1 = ql - al
        if m1>= config.QADIFF_THRESHOLD or m1 <= (-1)*config.QADIFF_THRESHOLD or ql > maxInput or al > maxOutput:
            continue
        if i in testIds:
            filenamesWithPathList[2].write(questions[i] + '\n')
            filenamesWithPathList[3].write(answers[i] + '\n')
        else:
            filenamesWithPathList[0].write(questions[i] + '\n')
            filenamesWithPathList[1].write(answers[i] + '\n')

    for file in filenamesWithPathList:
        file.close()

def makeOutputDirectory(path):
    """ 
    Create a directory to output all processed data. 
    If directory exists do not create and return
    """
    try:
        os.mkdir(path)
    except OSError:
        pass

def lineTokenizer(bline):
    """  
    The data contains some markers that need to be removed
    In addition we remove all apostrophies
    
    ie don't becomes dont i'm becomes im and so forth
    
    Only words are extracted and punctuation is 
    disregarded.
    """
    line = str(bline)
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('<b>', '', line)
    line = re.sub('</b>', '', line)
    line = re.sub('<i>', '', line)
    line = re.sub('</i>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    line = re.sub('\.\.\.[\.]+','\.\.\.', line)
    line = re.sub('[\?]+','\?', line)
    line = re.sub('[\!]+','\!', line)
    line = re.sub('[,]+',',', line)
    if config.USEAPO==False:
        line = re.sub('\'','',line)
    else:
        line = re.sub('[\']+','\'',line)
    line = re.sub('\"','',line)
    line = re.sub('#',' number ',line)
    line = re.sub('19',' nineteen ',line)
    line = re.sub('1',' one ',line)
    line = re.sub('2',' two ',line)
    line = re.sub('3',' three ',line)
    line = re.sub('4',' four ',line)
    line = re.sub('5',' five ',line)
    line = re.sub('6',' six ',line)
    line = re.sub('7',' seven ',line)
    line = re.sub('8',' eight ',line)
    line = re.sub('9',' nine ',line)
    line = re.sub('0',' zero ',line)
    
    
    
    """
    All words found in a line will be added to this list
    """
    words = []
    
    """
    The next regular expression defines a word
    We accept only words formed from letters
    Before we proceed we change all upper cases
    to lower case. 
    
    This reduces the dictionary size as 
    common nouns are capitalized at the start
    of a sentence.
    
    It also simplifies the word definition
    as we now have to deal only with lower case
    letters.
    
    """
    _WORD_EXPRESSION = re.compile(config.REGRULE)
    
    for fragment in line.strip().lower().split():
        for token in re.findall(_WORD_EXPRESSION, fragment):
            if not token:
                continue
            words.append(token)
            
    return words


def buildVocabulary(encFilename, decFilename, encTestFilename, decTestFilename):
    """
    We use a single vocabulary file for all files
    All sentences are indexed against this file 
    The vocabulary filename is configurable in
    config.py
    There are 4 files to extract vocabulary from:
    test.enc, test.dec, train.enc and train.dec
    """
    inEncPath = os.path.join(config.PROCESSED_PATH, encFilename)
    inDecPath = os.path.join(config.PROCESSED_PATH, decFilename)
    
    inEncTestPath = os.path.join(config.PROCESSED_PATH, encTestFilename)
    inDecTestPath = os.path.join(config.PROCESSED_PATH, decTestFilename)
    
    fileList = [inEncPath, inDecPath, inEncTestPath, inDecTestPath]
    
    outPath = os.path.join(config.PROCESSED_PATH, config.VOCAB_FILE)

    vocab = {}
    for file in fileList:      
        with open(file, 'r') as file:
            for line in file.readlines():
                for token in lineTokenizer(line):
                    if not token in vocab:
                        vocab[token] = 0
                    vocab[token] += 1
        file.close()

    sortedVocab = sorted(vocab, key=vocab.get, reverse=True)
    
    with open(outPath, 'w') as file:
        for spid in config.SPECIAL_ID:
            file.write(config.SPECIAL_SEQ[spid] + '\n')

        index = len(config.SPECIAL_ID)
        for word in sortedVocab:
            if vocab[word] < config.THRESHOLD:
                break
            file.write(word + '\n')
            index += 1
        file.close()
    

def loadVocabulary(vocab_path):
    with open(vocab_path, 'r') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}

def sentence2ID(vocab, line):
    return [vocab.get(token, vocab[config.SPECIAL_SEQ[config.UNK_ID]]) for token in lineTokenizer(line)]

def vectorDiff(a, b):
    return [a[i]-b[i] for i,_ in enumerate(a)]

def buildIDFiles(trainFile, testFile):
    """ 
    Convert all the tokens into their corresponding
    index in the vocabulary.
    
    """
    _, vocab = loadVocabulary(os.path.join(config.PROCESSED_PATH, config.VOCAB_FILE))
    
    fileList = [trainFile, testFile]
    
    encTrainLen = []
    decTrainLen = []
    encTestLen = []
    decTestLen = []
    
    for file in fileList:   
        inPath =  file + config.ENCODER
        outPath = file + config.IDS + config.ENCODER
    
        inFile = open(os.path.join(config.PROCESSED_PATH, inPath), 'r')
        outFile = open(os.path.join(config.PROCESSED_PATH, outPath), 'w')
        
        lines = inFile.read().splitlines()
        for line in lines:
            ids = []
            lineIds = sentence2ID(vocab, line)
            if file == trainFile:
                encTrainLen.append(len(lineIds))
            else:
                encTestLen.append(len(lineIds))
            ids.extend(lineIds)
            outFile.write(' '.join(str(id_) for id_ in ids) + '\n')
        inFile.close()
        outFile.close()

    for file in fileList:   
        inDecPath =  file + config.DECODER
        outDecPath = file + config.IDS + config.DECODER
    
        inDecFile = open(os.path.join(config.PROCESSED_PATH, inDecPath), 'r')
        outDecFile = open(os.path.join(config.PROCESSED_PATH, outDecPath), 'w')
        
        lines = inDecFile.read().splitlines()
        for line in lines:
            ids = [vocab[config.SPECIAL_SEQ[config.START_ID]]]
            lineIds = sentence2ID(vocab, line)
            if file == trainFile:
                decTrainLen.append(len(lineIds))
            else:
                decTestLen.append(len(lineIds))
            ids.extend(lineIds)
            ids.append(vocab[config.SPECIAL_SEQ[config.END_ID]])
            outDecFile.write(' '.join(str(id_) for id_ in ids) + '\n')
        inDecFile.close()
        outDecFile.close()
        

#     plt.subplot(211)
#     plt.hist(vectorDiff(encTrainLen, decTrainLen), 200)
#     plt.title('train set hist enc-dec')
#     
#     plt.subplot(212)
#     plt.hist(vectorDiff(encTestLen, decTestLen), 200)
#     plt.title('test set hist enc-dec')  
#     plt.show()


def loadData(encFilename, decFilename, maxTrainingSize=None):
    encodeFile = open(os.path.join(config.PROCESSED_PATH, encFilename), 'r')
    decodeFile = open(os.path.join(config.PROCESSED_PATH, decFilename), 'r')
    encode, decode = encodeFile.readline(), decodeFile.readline()
    dataBuckets = [[] for _ in config.BUCKETS]
    i = 0
    while encode and decode:
        if (i + 1) % config.DATABUCKET == 0:
            print("Bucketing conversation number", i)
        encodeIds = [int(id_) for id_ in encode.split()]
        decodeIds = [int(id_) for id_ in decode.split()]
        for bucketId, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encodeIds) <= encode_max_size and len(decodeIds) <= decode_max_size:
                dataBuckets[bucketId].append([encodeIds, decodeIds])
                break
        encode, decode = encodeFile.readline(), decodeFile.readline()
        i += 1
    return dataBuckets

def _padInput(input_, size):
    return input_ + [config.PAD_ID] * (size - len(input_))

def _reshapeBatch(inputs, size, batchSize):
    """ 
    Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batchInputs = []
    for lengthId in range(size):
        batchInputs.append(np.array([inputs[batchId][lengthId]
                                    for batchId in range(batchSize)], dtype=np.int32))
    return batchInputs


def getBatch(dataBucket, bucketId, batchSize=1):
    """ 
    Return one batch to feed into the model 
    Get a random batch of data from the specified bucket, prepare for step.
    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.
    Args:
      dataBucket: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucketId: integer, which bucket to get the batch for.
    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    # only pad to the max length of the bucket
    encoderSize, decoderSize = config.BUCKETS[bucketId]
    encoderInputs, decoderInputs = [], []

    for _ in range(batchSize):
        encoderInput, decoderInput = random.choice(dataBucket)
        # pad both encoder and decoder, reverse the encoder
        encoderInputs.append(list(reversed(_padInput(encoderInput, encoderSize))))
        decoderInputs.append(_padInput(decoderInput, decoderSize))

    # now we create batch-major vectors from the data selected above.
    batchEncoderInputs = _reshapeBatch(encoderInputs, encoderSize, batchSize)
    batchDecoderInputs = _reshapeBatch(decoderInputs, decoderSize, batchSize)

    # create decoder_masks to be 0 for decoders that are padding.
    batchMasks = []
    for lengthId in range(decoderSize):
        batch_mask = np.ones(batchSize, dtype=np.float32)
        for batchId in range(batchSize):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoderInput shifted by 1 forward.
            if lengthId < decoderSize - 1:
                target = decoderInputs[batchId][lengthId + 1]
            if lengthId == decoderSize - 1 or target == config.PAD_ID:
                batch_mask[batchId] = 0.0
        batchMasks.append(batch_mask)
    return batchEncoderInputs, batchDecoderInputs, batchMasks

def prepareRawData():
    print('Building Q and A sets from raw data...')
    questions, answers = questionsAndAnswers(getLines(), getConversations())
    prepareDataset(questions, answers)

def processData():
    print('Building vocabulary ...')
    buildVocabulary(config.TRAINFILE+config.ENCODER, config.TRAINFILE+config.DECODER, config.TESTFILE+config.ENCODER, config.TESTFILE+config.DECODER)
    print('Building train and test data sets ...')
    buildIDFiles(config.TRAINFILE, config.TESTFILE)

if __name__ == '__main__':
    prepareRawData()
    processData()