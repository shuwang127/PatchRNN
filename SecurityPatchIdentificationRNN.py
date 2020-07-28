'''
    SecurityPatchIdentificationRNN: Security Patch Identification using RNN model.
    Developer: Shu Wang
    Date: 2020-07-23
    Version: S2020.07.23-V3
    Description: patch identification using both commit messages and normalized diff code.
    File Structure:
    SecurityPatchIdentificationRNN
        |-- analysis                                # task analysis.
        |-- data                                    # data storage.
                |-- negatives                           # negative samples.
                |-- positives                           # positive samples.
                |-- security_patch                      # positive samples. (official)
        |-- temp                                    # temporary stored variables.
                |-- data.npy                            # raw data. (important)
                |-- props.npy                           # properties of diff code. (important)
                |-- msgs.npy                            # commit messages. (important)
                |-- ...                                 # other temporary files. (trivial)
        |-- SecurityPatchIdentificationRNN.ipynb    # main entrance. (Google Colaboratory)
        |-- SecurityPatchIdentificationRNN.py       # main entrance. (Local)
    Usage:
        python SecurityPatchIdentificationRNN.py
    Dependencies:
        clang >= 6.0.0.2
        torch >= 1.2.0+cu92
        nltk  >= 3.3
'''

# environment settings.
_COLAB_ = 0 # 0 : Local environment.
            # 1 : Google Colaboratory.

# dependencies.
import os
os.system('pip install clang')
import re
import gc
import math
import random
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import clang.cindex
import clang.enumerations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
from sklearn.metrics import accuracy_score

# file path.
rootPath = './drive/My Drive/Colab Notebooks/' if (_COLAB_) else './'
dataPath = rootPath + '/data/'
sDatPath = dataPath + '/security_patch/'
pDatPath = dataPath + '/positives/'
nDatPath = dataPath + '/negatives/'
tempPath = rootPath + '/temp/'

# hyper-parameters. (affect GPU memory)
_DiffEmbedDim_  = 128       # 128
_DiffMaxLen_    = 600       # 200(0.7), 314(0.8), 609(0.9), 1100(0.95), 2200(0.98), 3289(0.99), 5000(0.995), 10000(0.9997)
_TRnnHidSiz_    = 32        # 32
_MsgEmbedDim_   = 128       # 128
_MsgMaxLen_     = 200       # 54(0.9), 78(0.95), 130(0.98), 187(0.99), 268(0.995), 356(0.998), 516(0.999), 1434(1)
_MRnnHidSiz_    = 16        # 16
# hyper-parameters. (affect training speed)
_TRnnBatchSz_   = 128       # 128
_TRnnLearnRt_   = 0.0001    # 0.0001
_MRnnBatchSz_   = 128       # 128
_MRnnLearnRt_   = 0.0001    # 0.0001
# hyper-parameters. (unnecessary to modify)
_DiffExtraDim_  = 2         # 2
_TRnnHidLay_    = 1         # 1
_TRnnMaxEpoch_  = 1000      # 1000
_TRnnPerEpoch_  = 1         # 1
_TRnnJudEpoch_  = 10        # 10
_MRnnHidLay_    = 1         # 1
_MRnnMaxEpoch_  = 1000      # 1000
_MRnnPerEpoch_  = 1         # 1
_MRnnJudEpoch_  = 10        # 10

# control
_DEBUG_ = 0 # 0 : release
            # 1 : debug
_LOCK_ = 0  # 0 : unlocked - create random split sets.
            # 1 : locked   - use the saved split sets.
_MODEL_ = 0 # 0 : unlocked - train a new model.
            # 1 : locked   - load the saved model.

def demoTextRNN():
    '''
    demo program of using diff code to identify patches.
    '''

    # load data.
    if (not os.path.exists(tempPath + '/data.npy')): # | (not _DEBUG_)
        dataLoaded = ReadData()
    else:
        dataLoaded = np.load(tempPath + '/data.npy', allow_pickle=True)
        print('[INFO] <ReadData> Load ' + str(len(dataLoaded)) + ' raw data from ' + tempPath + '/data.npy.')

    # get the diff file properties.
    if (not os.path.exists(tempPath + '/props.npy')):
        diffProps = GetDiffProps(dataLoaded)
    else:
        diffProps = np.load(tempPath + '/props.npy', allow_pickle=True)
        print('[INFO] <GetDiffProps> Load ' + str(len(diffProps)) + ' diff property data from ' + tempPath + '/props.npy.')

    # get the diff token vocabulary.
    diffVocab, diffMaxLen = GetDiffVocab(diffProps)
    # get the max diff length.
    diffMaxLen = _DiffMaxLen_ if (diffMaxLen > _DiffMaxLen_) else diffMaxLen
    # get the diff token dictionary.
    diffDict = GetDiffDict(diffVocab)
    # get pre-trained weights for embedding layer.
    diffPreWeights = GetDiffEmbed(diffDict, _DiffEmbedDim_)
    # get the mapping for feature data and labels.
    diffData, diffLabels = GetDiffMapping(diffProps, diffMaxLen, diffDict)
    # change the tokentypes into one-hot vector.
    diffData = UpdateTokenTypes(diffData)

    # split data into rest/test dataset.
    dataRest, labelRest, dataTest, labelTest = SplitData(diffData, diffLabels, 'test', rate=0.2)
    # split data into train/valid dataset.
    dataTrain, labelTrain, dataValid, labelValid = SplitData(dataRest, labelRest, 'valid', rate=0.2)
    print('[INFO] <main> Get ' + str(len(dataTrain)) + ' TRAIN data, ' + str(len(dataValid)) + ' VALID data, '
          + str(len(dataTest)) + ' TEST data. (Total: ' + str(len(dataTrain)+len(dataValid)+len(dataTest)) + ')')

    # TextRNNTrain
    if (_MODEL_) & (os.path.exists(tempPath + '/model_TextRNN.pth')):
        preWeights = torch.from_numpy(diffPreWeights)
        model = MsgRNN(preWeights, hiddenSize=_TRnnHidSiz_, hiddenLayers=_TRnnHidLay_)
        model.load_state_dict(torch.load(tempPath + '/model_TextRNN.pth'))
    else:
        model = TextRNNTrain(dataTrain, labelTrain, dataValid, labelValid, preWeights=diffPreWeights,
                             batchsize=_TRnnBatchSz_, learnRate=_TRnnLearnRt_, dTest=dataTest, lTest=labelTest)

    # TextRNNTest
    predictions, accuracy = TextRNNTest(model, dataTest, labelTest, batchsize=_TRnnBatchSz_)
    _, confusion = OutputEval(predictions, labelTest, 'TextRNN')

    return

def ReadData():
    '''
    Read data from the files.
    :return: data - a set of commit message, diff code, and labels.
    [[['', ...], [['', ...], ['', ...], ...], 0/1], ...]
    '''

    def ReadCommitMsg(filename):
        '''
        Read commit message from a file.
        :param filename: file name (string).
        :return: commitMsg - commit message.
        ['line', 'line', ...]
        '''

        fp = open(filename, encoding='utf-8', errors='ignore')  # get file point.
        lines = fp.readlines()  # read all lines.
        #numLines = len(lines)   # get the line number.
        #print(lines)

        # initialize commit message.
        commitMsg = []
        # get the wide range of commit message.
        for line in lines:
            if line.startswith('diff --git'):
                break
            else:
                commitMsg.append(line)
        #print(commitMsg)
        # process the head of commit message.
        while (1):
            headMsg = commitMsg[0]
            if (headMsg.startswith('From') or headMsg.startswith('Date:') or headMsg.startswith('Subject:')
                    or headMsg.startswith('commit') or headMsg.startswith('Author:')):
                commitMsg.pop(0)
            else:
                break
        #print(commitMsg)
        # process the tail of commit message.
        dashLines = [i for i in range(len(commitMsg))
                     if commitMsg[i].startswith('---')]  # finds all lines start with ---.
        if (len(dashLines)):
            lnum = dashLines[-1]  # last line number of ---
            marks = [1 if (' file changed, ' in commitMsg[i] or ' files changed, ' in commitMsg[i]) else 0
                     for i in range(lnum, len(commitMsg))]
            if (sum(marks)):
                for i in reversed(range(lnum, len(commitMsg))):
                    commitMsg.pop(i)
        #print(commitMsg)

        #msgShow = ''
        #for i in range(len(commitMsg)):
        #    msgShow += commitMsg[i]
        #print(msgShow)

        return commitMsg

    def ReadDiffLines(filename):
        '''
        Read diff code from a file.
        :param filename:  file name (string).
        :return: diffLines - diff code.
        [['line', ...], ['line', ...], ...]
        '''

        fp = open(filename, encoding='utf-8', errors='ignore')  # get file point.
        lines = fp.readlines()  # read all lines.
        numLines = len(lines)  # get the line number.
        # print(lines)

        atLines = [i for i in range(numLines) if lines[i].startswith('@@ ')]  # find all lines start with @@.
        atLines.append(numLines)
        # print(atLines)

        diffLines = []
        for nh in range(len(atLines) - 1):  # find all hunks.
            # print(atLines[nh], atLines[nh + 1])
            hunk = []
            for nl in range(atLines[nh] + 1, atLines[nh + 1]):
                # print(lines[nl], end='')
                if lines[nl].startswith('diff --git '):
                    break
                else:
                    hunk.append(lines[nl])
            diffLines.append(hunk)
            # print(hunk)
        # print(diffLines)
        # print(len(diffLines))

        # process the last hunk.
        lastHunk = diffLines[-1]
        numLastHunk = len(lastHunk)
        dashLines = [i for i in range(numLastHunk) if lastHunk[i].startswith('--')]
        if (len(dashLines)):
            lnum = dashLines[-1]
            for i in reversed(range(lnum, numLastHunk)):
                lastHunk.pop(i)
        # print(diffLines)
        # print(len(diffLines))

        return diffLines

    # create temp folder.
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    fp = open(tempPath + 'filelist.txt', 'w')

    # initialize data.
    data = []
    # read security patch data.
    for root, ds, fs in os.walk(sDatPath):
        for file in fs:
            filename = os.path.join(root, file).replace('\\', '/')
            fp.write(filename + '\n')
            commitMsg = ReadCommitMsg(filename)
            diffLines = ReadDiffLines(filename)
            data.append([commitMsg, diffLines, 1])

    # read positive data.
    for root, ds, fs in os.walk(pDatPath):
        for file in fs:
            filename = os.path.join(root, file).replace('\\', '/')
            fp.write(filename + '\n')
            commitMsg = ReadCommitMsg(filename)
            diffLines = ReadDiffLines(filename)
            data.append([commitMsg, diffLines, 1])

    # read negative data.
    for root, ds, fs in os.walk(nDatPath):
        for file in fs:
            filename = os.path.join(root, file).replace('\\', '/')
            fp.write(filename + '\n')
            commitMsg = ReadCommitMsg(filename)
            diffLines = ReadDiffLines(filename)
            data.append([commitMsg, diffLines, 0])
    fp.close()

    #print(len(dataLoaded))
    #print(len(dataLoaded[0]))
    #print(dataLoaded)
    # [[['a', 'b', 'c', ], [['', '', '', ], ['', '', '', ], ], 0/1], ]
    # sample = dataLoaded[i]
    # commitMsg = dataLoaded[i][0]
    # diffLines = dataLoaded[i][1]
    # label = dataLoaded[i][2]
    # diffHunk = dataLoaded[i][1][j]

    # save dataLoaded.
    if not os.path.exists(tempPath + '/data.npy'):
        np.save(tempPath + '/data.npy', data, allow_pickle=True)
        print('[INFO] <ReadData> Save ' + str(len(data)) + ' raw data to ' + tempPath + '/data.npy.')

    return data

def GetDiffProps(data):
    '''
    Get the properties of the code in diff files.
    :param data: [[[line, , ], [[line, , ], [line, , ], ...], 0/1], ...]
    :return: props - [[[tokens], [nums], [nums], 0/1], ...]
    '''

    def RemoveSign(line):
        '''
        Remove the sign (+/-) in the first character.
        :param line: a code line.
        :return: process line.
        '''

        return ' ' + line[1:] if (line[0] == '+') or (line[0] == '-') else line

    def GetClangTokens(line):
        '''
        Get the tokens of a line with the Clang tool.
        :param line: a code line.
        :return: tokens - ['tk', 'tk', ...] ('tk': string)
                 tokenTypes - [tkt, tkt, ...] (tkt: 1, 2, 3, 4, 5)
                 diffTypes - [dft, dft, ...] (dft: -1, 0, 1)
        '''

        # remove non-ascii
        line = line.encode("ascii", "ignore").decode()

        # defination.
        tokenClass = [clang.cindex.TokenKind.KEYWORD,      # 1
                      clang.cindex.TokenKind.IDENTIFIER,   # 2
                      clang.cindex.TokenKind.LITERAL,      # 3
                      clang.cindex.TokenKind.PUNCTUATION,  # 4
                      clang.cindex.TokenKind.COMMENT]      # 5
        tokenDict = {cls: index + 1 for index, cls in enumerate(tokenClass)}
        #print(tokenDict)

        # initialize.
        tokens = []
        tokenTypes = []
        diffTypes = []

        # clang sparser.
        idx = clang.cindex.Index.create()
        tu = idx.parse('tmp.cpp', args=['-std=c++11'], unsaved_files=[('tmp.cpp', RemoveSign(line))], options=0)
        for t in tu.get_tokens(extent=tu.cursor.extent):
            #print(t.kind, t.spelling, t.location)
            tokens.append(t.spelling)
            tokenTypes.append(tokenDict[t.kind])
            diffTypes.append(1 if (line[0] == '+') else -1 if (line[0] == '-') else 0)
        #print(tokens)
        #print(tokenTypes)
        #print(diffTypes)

        return tokens, tokenTypes, diffTypes

    def GetWordTokens(line):
        '''
        Get the word tokens from a code line.
        :param line: a code line.
        :return: tokens - ['tk', 'tk', ...] ('tk': string)
        '''

        tknzr = TweetTokenizer()
        tokens = tknzr.tokenize(RemoveSign(line))
        return tokens

    def GetString(lines):
        '''
        Get the strings from the diff code
        :param lines: diff code.
        :return: lineStr - All the diff lines.
                 lineStrB - The before-version code lines.
                 lineStrA - The after-version code lines.
        '''

        lineStr = ''
        lineStrB = ''
        lineStrA = ''
        for hunk in lines:
            for line in hunk:
                # all lines.
                lineStr += RemoveSign(line)
                # all Before lines.
                lineStrB += RemoveSign(line) if line[0] != '+' else ''
                # all After lines.
                lineStrA += RemoveSign(line) if line[0] != '-' else ''

        return lineStr, lineStrB, lineStrA

    def GetDiffTokens(lines):
        '''
        Get the tokens for the diff lines.
        :param lines: the diff code.
        :return: tokens - tokens ['tk', 'tk', ...] ('tk': string)
                 tokenTypes - token types [tkt, tkt, ...] (tkt: 1, 2, 3, 4, 5)
                 diffTypes - diff types [dft, dft, ...] (dft: -1, 0, 1)
        '''

        # initialize.
        tokens = []
        tokenTypes = []
        diffTypes = []

        # for each line of lines.
        for hunk in lines:
            for line in hunk:
                #print(line, end='')
                tk, tkT, dfT = GetClangTokens(line)
                tokens.extend(tk)
                tokenTypes.extend(tkT)
                diffTypes.extend(dfT)
                #print('-----------------------------------------------------------------------')
        #print(tokens)
        #print(tokenTypes)
        #print(diffTypes)

        return tokens, tokenTypes, diffTypes

    #lines = data[0][1]
    #print(lines)
    #hunk = data[0][1][0]
    #print(hunk)
    #line = data[0][1][0][0]
    #print(line)

    # for each sample data[n].
    numData = len(data)
    props = []
    for n in range(numData):
        # get the lines of the diff file.
        diffLines = data[n][1]
        # properties.
        tk, tkT, dfT = GetDiffTokens(diffLines)
        label = data[n][2]
        prop = [tk, tkT, dfT, label]
        #print(prop)
        props.append(prop)
        print(n)

    # save props.
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    if not os.path.exists(tempPath + '/props.npy'):
        np.save(tempPath + '/props.npy', props, allow_pickle=True)
        print('[INFO] <GetDiffProps> Save ' + str(len(props)) + ' diff property data to ' + tempPath + '/props.npy.')

    return props

def GetDiffVocab(props):
    '''
    Get the vocabulary of diff tokens
    :param props - the features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    :return: vocab - the vocabulary of diff tokens. ['tk', 'tk', ...]
             maxLen - the max length of a diff code.
    '''

    # create temp folder.
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    fp = open(tempPath + 'difflen.csv', 'w')

    # get the whole tokens and the max diff length.
    tokens = []
    maxLen = 0

    # for each sample.
    for item in props:
        tokens.extend(item[0])
        maxLen = len(item[0]) if (len(item[0]) > maxLen) else maxLen
        fp.write(str(len(item[0])) + '\n')
    fp.close()

    # remove duplicates and get vocabulary.
    vocab = {}.fromkeys(tokens)
    vocab = list(vocab.keys())

    # print.
    print('[INFO] <GetDiffVocab> There are ' + str(len(vocab)) + ' diff vocabulary tokens. (except \'<pad>\')')
    print('[INFO] <GetDiffVocab> The max diff length is ' + str(maxLen) + ' tokens. (hyperparameter: _DiffMaxLen_ = ' + str(_DiffMaxLen_) + ')')

    return vocab, maxLen

def GetDiffDict(vocab):
    '''
    Get the dictionary of diff vocabulary.
    :param vocab: the vocabulary of diff tokens. ['tk', 'tk', ...]
    :return: tokenDict - the dictionary of diff vocabulary.
    {'tk': 1, 'tk': 2, ..., 'tk': N, '<pad>': 0}
    '''

    # get token dict from vocabulary.
    tokenDict = {token: (index+1) for index, token in enumerate(vocab)}
    tokenDict['<pad>'] = 0

    # print.
    print('[INFO] <GetDiffDict> Create dictionary for ' + str(len(tokenDict)) + ' diff vocabulary tokens. (with \'<pad>\')')

    return tokenDict

def GetDiffEmbed(tokenDict, embedSize):
    '''
    Get the pre-trained weights for embedding layer from the dictionary of diff vocabulary.
    :param tokenDict: the dictionary of diff vocabulary.
    {'tk': 0, 'tk': 1, ..., '<pad>': N}
    :param embedSize: the dimension of the embedding vector.
    :return: preWeights - the pre-trained weights for embedding layer.
    [[n, ...], [n, ...], ...]
    '''

    # number of the vocabulary tokens.
    numTokens = len(tokenDict)

    # initialize the pre-trained weights for embedding layer.
    preWeights = np.zeros((numTokens, embedSize))
    for index in range(numTokens):
        preWeights[index] = np.random.normal(size=(embedSize,))
    print('[INFO] <GetDiffEmbed> Create pre-trained embedding weights with ' + str(len(preWeights)) + ' * ' + str(len(preWeights[0])) + ' matrix.')

    # save preWeights.
    if not os.path.exists(tempPath + '/preWeights.npy'):
        np.save(tempPath + '/preWeights.npy', preWeights, allow_pickle=True)
        print('[INFO] <GetDiffEmbed> Save the pre-trained weights of embedding layer to ' + tempPath + '/preWeights.npy.')

    return preWeights

def GetDiffMapping(props, maxLen, tokenDict):
    '''
    Map the feature data into indexed data.
    :param props: the features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    :param maxLen: the max length of a diff code.
    :param tokenDict: the dictionary of diff vocabulary.
    {'tk': 1, 'tk': 2, ..., 'tk': N, '<pad>': 0}
    :return: np.array(data) - feature data.
             [[[n, {0~5}, {-1~1}], ...], ...]
             np.array(labels) - labels.
             [[0/1], ...]
    '''

    def PadList(dList, pad, length):
        '''
        Pad the list data to a fixed length.
        :param dList: the list data - [ , , ...]
        :param pad: the variable used to pad.
        :param length: the fixed length.
        :return: dList - padded list data. [ , , ...]
        '''

        if len(dList) <= length:
            dList.extend(pad for i in range(length - len(dList)))
        elif len(dList) > length:
            dList = dList[0:length]

        return dList

    # initialize the data and labels.
    data = []
    labels = []

    # for each sample.
    for item in props:
        # initialize sample.
        sample = []

        # process token.
        tokens = item[0]
        tokens = PadList(tokens, '<pad>', maxLen)
        tokens2index = []
        for tk in tokens:
            tokens2index.append(tokenDict[tk])
        sample.append(tokens2index)
        # process tokenTypes.
        tokenTypes = item[1]
        tokenTypes = PadList(tokenTypes, 0, maxLen)
        sample.append(tokenTypes)
        # process diffTypes.
        diffTypes = item[2]
        diffTypes = PadList(diffTypes, 0, maxLen)
        sample.append(diffTypes)

        # process sample.
        sample = np.array(sample).T
        data.append(sample)
        # process label.
        label = item[3]
        labels.append([label])

    if _DEBUG_:
        print('[DEBUG] data:')
        print(data[0:3])
        print('[DEBUG] labels:')
        print(labels[0:3])

    # print.
    print('[INFO] <GetDiffMapping> Create ' + str(len(data)) + ' feature data with ' + str(len(data[0])) + ' * ' + str(len(data[0][0])) + ' matrix.')
    print('[INFO] <GetDiffMapping> Create ' + str(len(labels)) + ' labels with 1 * 1 matrix.')

    # save files.
    if (not os.path.exists(tempPath + '/ndata_' + str(maxLen) + '.npy')) \
            | (not os.path.exists(tempPath + '/nlabels_' + str(maxLen) + '.npy')):
        np.save(tempPath + '/ndata_' + str(maxLen) + '.npy', data, allow_pickle=True)
        print('[INFO] <GetDiffMapping> Save the mapped numpy data to ' + tempPath + '/ndata_' + str(maxLen) + '.npy.')
        np.save(tempPath + '/nlabels_' + str(maxLen) + '.npy', labels, allow_pickle=True)
        print('[INFO] <GetDiffMapping> Save the mapped numpy labels to ' + tempPath + '/nlabels_' + str(maxLen) + '.npy.')

    return np.array(data), np.array(labels)

def UpdateTokenTypes(data):
    '''
    Update the token type in the feature data into one-hot vector.
    :param data: feature data. [[[n, {0~5}, {-1~1}], ...], ...]
    :return: np.array(newData). [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    '''

    newData = []
    # for each sample.
    for item in data:
        # get the transpose of props.
        itemT = item.T
        # initialize new sample.
        newItem = []
        newItem.append(itemT[0])
        newItem.extend(np.zeros((5, len(item)), dtype=int))
        newItem.append(itemT[2])
        # assign the new sample.
        for i in range(len(item)):
            tokenType = itemT[1][i]
            if (tokenType):
                newItem[tokenType][i] = 1
        # get the transpose of new sample.
        newItem = np.array(newItem).T
        # append new sample.
        newData.append(newItem)

    if _DEBUG_:
        print('[DEBUG] newData:')
        print(newData[0:3])

    # print.
    print('[INFO] <UpdateTokenTypes> Update ' + str(len(newData)) + ' feature data with ' + str(len(newData[0])) + ' * ' + str(len(newData[0][0])) + ' matrix.')

    # save files.
    if (not os.path.exists(tempPath + '/newdata_' + str(len(newData[0])) + '.npy')):
        np.save(tempPath + '/newdata_' + str(len(newData[0])) + '.npy', newData, allow_pickle=True)
        print('[INFO] <UpdateTokenTypes> Save the mapped numpy data to ' + tempPath + '/newdata_' + str(len(newData[0])) + '.npy.')

    # change marco.
    global _DiffExtraDim_
    _DiffExtraDim_ = 6

    return np.array(newData)

def SplitData(data, labels, setType, rate=0.2):
    '''
    Split the data and labels into two sets with a specific rate.
    :param data: feature data.
    [[[n, {0~5}, {-1~1}], ...], ...]
    [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :param labels: labels. [[0/1], ...]
    :param setType: the splited dataset type.
    :param rate: the split rate. 0 ~ 1
    :return: dsetRest - the rest dataset.
             lsetRest - the rest labels.
             dset - the splited dataset.
             lset - the splited labels.
    '''

    # set parameters.
    setType = setType.upper()
    numData = len(data)
    num = math.floor(numData * rate)

    # get the random data list.
    if (os.path.exists(tempPath + '/split_' + setType + '.npy')) & (_LOCK_):
        dataList = np.load(tempPath + '/split_' + setType + '.npy')
    else:
        dataList = list(range(numData))
        random.shuffle(dataList)
        np.save(tempPath + '/split_' + setType + '.npy', dataList, allow_pickle=True)

    # split data.
    dset = data[dataList[0:num]]
    lset = labels[dataList[0:num]]
    dsetRest = data[dataList[num:]]
    lsetRest = labels[dataList[num:]]

    # print.
    setTypeRest = 'TRAIN' if (setType == 'VALID') else 'REST'
    print('[INFO] <SplitData> Split data into ' + str(len(dsetRest)) + ' ' + setTypeRest
          + ' dataset and ' + str(len(dset)) + ' ' + setType + ' dataset. (Total: '
          + str(len(dsetRest) + len(dset)) + ', Rate: ' + str(int(rate * 100)) + '%)')

    return dsetRest, lsetRest, dset, lset

class TextRNN(nn.Module):
    '''
    TextRNN : convert a text data into a predicted label.
    '''

    def __init__(self, preWeights, hiddenSize=32, hiddenLayers=1):
        '''
        define each layer in the network model.
        :param preWeights: tensor pre-trained weights for embedding layer.
        :param hiddenSize: node number in the hidden layer.
        :param hiddenLayers: number of hidden layer.
        '''

        super(TextRNN, self).__init__()
        # parameters.
        class_num = 2
        vocabSize, embedDim = preWeights.size()
        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocabSize, embedding_dim=embedDim)
        self.embedding.load_state_dict({'weight': preWeights})
        self.embedding.weight.requires_grad = True
        # LSTM Layer
        #_DiffExtraDim_ = 6
        if _DEBUG_: print(_DiffExtraDim_)
        self.lstm = nn.LSTM(input_size=embedDim+_DiffExtraDim_, hidden_size=hiddenSize, num_layers=hiddenLayers, bidirectional=True)
        # Fully-Connected Layer
        self.fc = nn.Linear(hiddenSize * hiddenLayers * 2, class_num)
        # Softmax non-linearity
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        convert inputs to predictions.
        :param x: input tensor. dimension: batch_size * diff_length * feature_dim.
        :return: self.softmax(final_out) - predictions.
        [[0.3, 0.7], [0.2, 0.8], ...]
        '''

        # x             batch_size * diff_length * feature_dim
        embeds = self.embedding(x[:,:,0])
        # embeds        batch_size * diff_length * embedding_dim
        features = x[:, :, 1:]
        # features      batch_size * diff_length * _DiffExtraDim_
        inputs = torch.cat((embeds.float(), features.float()), 2)
        # inputs        batch_size * diff_length * (embedding_dim + _DiffExtraDim_)
        inputs = inputs.permute(1, 0, 2)
        # inputs        diff_length * batch_size * (embedding_dim + _DiffExtraDim_)
        lstm_out, (h_n, c_n) = self.lstm(inputs)
        # lstm_out      diff_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        feature_map = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # feature_map   batch_size * (hidden_size * num_layers * direction_num)
        final_out = self.fc(feature_map)    # batch_size * class_num
        return self.softmax(final_out)      # batch_size * class_num

def TextRNNTrain(dTrain, lTrain, dValid, lValid, preWeights, batchsize=64, learnRate=0.001, dTest=None, lTest=None):
    '''
    Train the TextRNN model.
    :param dTrain: training data. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :param lTrain: training label. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :param dValid: validation data. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :param lValid: validation label. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :param preWeights: pre-trained weights for embedding layer.
    :param batchsize: number of samples in a batch.
    :param learnRate: learning rate.
    :param dTest: test data. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :param lTest: test label. [[[n, 0/1, 0/1, 0/1, 0/1, 0/1, {-1~1}], ...], ...]
    :return: model - the TextRNN model.
    '''

    # get the mark of the test dataset.
    if dTest is None: dTest = []
    if lTest is None: lTest = []
    markTest = 1 if (len(dTest)) & (len(lTest)) else 0

    # tensor data processing.
    xTrain = torch.from_numpy(dTrain).long().cuda()
    yTrain = torch.from_numpy(lTrain).long().cuda()
    xValid = torch.from_numpy(dValid).long().cuda()
    yValid = torch.from_numpy(lValid).long().cuda()
    if (markTest):
        xTest = torch.from_numpy(dTest).long().cuda()
        yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    train = torchdata.TensorDataset(xTrain, yTrain)
    trainloader = torchdata.DataLoader(train, batch_size=batchsize, shuffle=False)
    valid = torchdata.TensorDataset(xValid, yValid)
    validloader = torchdata.DataLoader(valid, batch_size=batchsize, shuffle=False)
    if (markTest):
        test = torchdata.TensorDataset(xTest, yTest)
        testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # get training weights.
    lbTrain = [item for sublist in lTrain.tolist() for item in sublist]
    weights = []
    for lb in range(2):
        weights.append(1 - lbTrain.count(lb) / len(lbTrain))
    lbWeights = torch.FloatTensor(weights).cuda()

    # build the model of recurrent neural network.
    preWeights = torch.from_numpy(preWeights)
    model = TextRNN(preWeights, hiddenSize=_TRnnHidSiz_, hiddenLayers=_TRnnHidLay_)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('[INFO] <TextRNNTrain> ModelType: TextRNN, HiddenNodes: %d, HiddenLayers: %d.' % (_TRnnHidSiz_, _TRnnHidLay_))
    print('[INFO] <TextRNNTrain> BatchSize: %d, LearningRate: %.4f, MaxEpoch: %d, PerEpoch: %d.' % (batchsize, learnRate, _TRnnMaxEpoch_, _TRnnPerEpoch_))
    # optimizing with stochastic gradient descent.
    optimizer = optim.Adam(model.parameters(), lr=learnRate)
    # seting loss function as mean squared error.
    criterion = nn.CrossEntropyLoss(weight=lbWeights)
    # memory
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # run on each epoch.
    accList = [0]
    for epoch in range(_TRnnMaxEpoch_):
        # training phase.
        model.train()
        lossTrain = 0
        predictions = []
        labels = []
        for iter, (data, label) in enumerate(trainloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # back propagation.
            optimizer.zero_grad()  # set the gradients to zero.
            yhat = model.forward(data)  # get output
            loss = criterion(yhat, label)
            loss.backward()
            optimizer.step()
            # statistic
            lossTrain += loss.item() * len(label)
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        lossTrain /= len(dTrain)
        # train accuracy.
        accTrain = accuracy_score(labels, predictions) * 100

        # validation phase.
        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for iter, (data, label) in enumerate(validloader):
                # data conversion.
                data = data.to(device)
                label = label.contiguous().view(-1)
                label = label.to(device)
                # forward propagation.
                yhat = model.forward(data)  # get output
                # statistic
                preds = yhat.max(1)[1]
                predictions.extend(preds.int().tolist())
                labels.extend(label.int().tolist())
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        # valid accuracy.
        accValid = accuracy_score(labels, predictions) * 100
        accList.append(accValid)

        # testing phase.
        if (markTest):
            model.eval()
            predictions = []
            labels = []
            with torch.no_grad():
                for iter, (data, label) in enumerate(testloader):
                    # data conversion.
                    data = data.to(device)
                    label = label.contiguous().view(-1)
                    label = label.to(device)
                    # forward propagation.
                    yhat = model.forward(data)  # get output
                    # statistic
                    preds = yhat.max(1)[1]
                    predictions.extend(preds.int().tolist())
                    labels.extend(label.int().tolist())
                    torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            # test accuracy.
            accTest = accuracy_score(labels, predictions) * 100

        # output information.
        if (0 == (epoch + 1) % _TRnnPerEpoch_):
            strAcc = '[Epoch {:03}] loss: {:.3}, train acc: {:.3f}%, valid acc: {:.3f}%.'.format(epoch + 1, lossTrain, accTrain, accValid)
            if (markTest):
                strAcc = strAcc[:-1] + ', test acc: {:.3f}%.'.format(accTest)
            print(strAcc)
        # save the best model.
        if (accList[-1] > max(accList[0:-1])):
            torch.save(model.state_dict(), tempPath + '/model_TextRNN.pth')
        # stop judgement.
        if (epoch >= _TRnnJudEpoch_) and (accList[-1] < min(accList[-1-_TRnnJudEpoch_:-1])):
            break

    # load best model.
    model.load_state_dict(torch.load(tempPath + '/model_TextRNN.pth'))
    print('[INFO] <TextRNNTrain> Finish training TextRNN model. (Best model: ' + tempPath + '/model_TextRNN.pth)')

    return model

def TextRNNTest(model, dTest, lTest, batchsize=64):
    '''
    Test the TextRNN model.
    :param model: deep learning model.
    :param dTest: test data.
    :param lTest: test label.
    :param batchsize: number of samples in a batch
    :return: predictions - predicted labels. [[0], [1], ...]
             accuracy - the total test accuracy. numeric
    '''

    # tensor data processing.
    xTest = torch.from_numpy(dTest).long().cuda()
    yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    test = torchdata.TensorDataset(xTest, yTest)
    testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # load the model of recurrent neural network.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # testing phase.
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for iter, (data, label) in enumerate(testloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # forward propagation.
            yhat = model.forward(data)  # get output
            # statistic
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # testing accuracy.
    accuracy = accuracy_score(labels, predictions) * 100
    predictions = [[item] for item in predictions]

    return predictions, accuracy

def OutputEval(predictions, labels, method=''):
    '''
    Output the evaluation results.
    :param predictions: predicted labels. [[0], [1], ...]
    :param labels: ground truth labels. [[1], [1], ...]
    :param method: method name. string
    :return: accuracy - the total accuracy. numeric
             confusion - confusion matrix [[1000, 23], [12, 500]]
    '''

    # evaluate the predictions with gold labels, and get accuracy and confusion matrix.
    def Evaluation(predictions, labels):

        # parameter settings.
        D = len(labels)
        cls = 2

        # get confusion matrix.
        confusion = np.zeros((cls, cls))
        for ind in range(D):
            nRow = int(predictions[ind][0])
            nCol = int(labels[ind][0])
            confusion[nRow][nCol] += 1

        # get accuracy.
        accuracy = 0
        for ind in range(cls):
            accuracy += confusion[ind][ind]
        accuracy /= D

        return accuracy, confusion

    # get accuracy and confusion matrix.
    accuracy, confusion = Evaluation(predictions, labels)
    precision = confusion[1][1] / (confusion[1][0] + confusion[1][1]) if (confusion[1][0] + confusion[1][1]) else 0
    recall = confusion[1][1] / (confusion[0][1] + confusion[1][1]) if (confusion[0][1] + confusion[1][1]) else 0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # output on screen and to file.
    print('       -------------------------------------------')
    print('       method           :  ' +  method) if len(method) else print('', end='')
    print('       accuracy  (ACC)  :  %.3f%%' % (accuracy * 100))
    print('       precision (P)    :  %.3f%%' % (precision * 100))
    print('       recall    (R)    :  %.3f%%' % (recall * 100))
    print('       F1 score  (F1)   :  %.3f' % (F1))
    print('       fall-out  (FPR)  :  %.3f%%' % (confusion[1][0] * 100 / (confusion[1][0] + confusion[0][0])))
    print('       miss rate (FNR)  :  %.3f%%' % (confusion[0][1] * 100 / (confusion[0][1] + confusion[1][1])))
    print('       confusion matrix :      (actual)')
    print('                           Neg         Pos')
    print('       (predicted) Neg     %-5d(TN)   %-5d(FN)' % (confusion[0][0], confusion[0][1]))
    print('                   Pos     %-5d(FP)   %-5d(TP)' % (confusion[1][0], confusion[1][1]))
    print('       -------------------------------------------')

    return accuracy, confusion

def demoCommitMsg():
    '''
    demo program of using commit message to identify patches.
    '''

    # load data.
    if (not os.path.exists(tempPath + '/data.npy')):  # | (not _DEBUG_)
        dataLoaded = ReadData()
    else:
        dataLoaded = np.load(tempPath + '/data.npy', allow_pickle=True)
        print('[INFO] <ReadData> Load ' + str(len(dataLoaded)) + ' raw data from ' + tempPath + '/data.npy.')

    # get the commit messages from data.
    if (not os.path.exists(tempPath + '/msgs.npy')):
        commitMsgs = GetCommitMsgs(dataLoaded)
    else:
        commitMsgs = np.load(tempPath + '/msgs.npy', allow_pickle=True)
        print('[INFO] <GetCommitMsg> Load ' + str(len(commitMsgs)) + ' commit messages from ' + tempPath + '/msgs.npy.')

    # get the message token vocabulary.
    msgVocab, msgMaxLen = GetMsgVocab(commitMsgs)
    # get the max msg length.
    msgMaxLen = _MsgMaxLen_ if (msgMaxLen > _MsgMaxLen_) else msgMaxLen
    # get the msg token dictionary.
    msgDict = GetMsgDict(msgVocab)
    # get pre-trained weights for embedding layer.
    msgPreWeights = GetMsgEmbed(msgDict, _MsgEmbedDim_)
    # get the mapping for feature data and labels.
    msgData, msgLabels = GetMsgMapping(commitMsgs, msgMaxLen, msgDict)
    # split data into rest/test dataset.
    mdataTrain, mlabelTrain, mdataTest, mlabelTest = SplitData(msgData, msgLabels, 'test', rate=0.2)

    # MsgRNNTrain
    if (_MODEL_) & (os.path.exists(tempPath + '/model_MsgRNN.pth')):
        preWeights = torch.from_numpy(msgPreWeights)
        model = TextRNN(preWeights, hiddenSize=_MRnnHidSiz_, hiddenLayers=_MRnnHidLay_)
        model.load_state_dict(torch.load(tempPath + '/model_MsgRNN.pth'))
    else:
        model = MsgRNNTrain(mdataTrain, mlabelTrain, mdataTest, mlabelTest, msgPreWeights,
                            batchsize=_MRnnBatchSz_, learnRate=_MRnnLearnRt_, dTest=mdataTest, lTest=mlabelTest)

    # MsgRNNTest
    predictions, accuracy = MsgRNNTest(model, mdataTest, mlabelTest, batchsize=_MRnnBatchSz_)
    _, confusion = OutputEval(predictions, mlabelTest, 'MsgRNN')

    return

def GetCommitMsgs(data):
    '''
    Get the commit messages in diff files.
    :param data: [[[line, , ], [[line, , ], [line, , ], ...], 0/1], ...]
    :return: msgs - [[[tokens], 0/1], ...]
    '''

    def GetMsgTokens(lines):
        '''
        Get the tokens from a commit message.
        :param lines: commit message. [line, , ]
        :return: tokensStem ['tk', , ]
        '''

        # concatenate lines.
        # get the string of commit message.
        msg = ''
        for line in lines:
            msg += line[:-1] + ' '
        #print(msg)

        # pre-process.
        # remove url.
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        msg = re.sub(pattern, ' ', msg)
        # remove independent numbers.
        pattern = r' \d+ '
        msg = re.sub(pattern, ' ', msg)
        # lower case capitalized words.
        pattern = r'([A-Z][a-z]+)'
        def LowerFunc(matched):
            return matched.group(1).lower()
        msg = re.sub(pattern, LowerFunc, msg)
        # remove footnote.
        patterns = ['signed-off-by:', 'reported-by:', 'reviewed-by:', 'acked-by:', 'found-by:', 'tested-by:', 'cc:']
        for pattern in patterns:
            index = msg.find(pattern)
            if (index > 0):
                msg = msg[:index]
        #print(msg)

        # clearance.
        # get the tokens.
        tknzr = TweetTokenizer()
        tokens = tknzr.tokenize(msg)
        # clear tokens that don't contain any english letter.
        for i in reversed(range(len(tokens))):
            if not (re.search('[a-z]', tokens[i])):
                tokens.pop(i)
        # clear tokens that are stopwords.
        for i in reversed(range(len(tokens))):
            if (tokens[i] in stopwords.words('english')):
                tokens.pop(i)
        pattern = re.compile("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
        for i in reversed(range(len(tokens))):
            if (pattern.findall(tokens[i])):
                tokens.pop(i)
        #print(tokens)

        # process tokens with stemming.
        porter = PorterStemmer()
        tokensStem = []
        for item in tokens:
            tokensStem.append(porter.stem(item))
        #print(tokensStem)

        return tokensStem

    # for each sample data[n].
    numData = len(data)
    msgs = []
    for n in range(numData):
        # get the lines of the commit message.
        commitMsg = data[n][0]
        mtk = GetMsgTokens(commitMsg)
        # get the label.
        label = data[n][2]
        #print([mtk, label])
        # append the message tokens.
        msgs.append([mtk, label])
        print(n)

    # save commit messages.
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    if not os.path.exists(tempPath + '/msgs.npy'):
        np.save(tempPath + '/msgs.npy', msgs, allow_pickle=True)
        print('[INFO] <GetCommitMsg> Save ' + str(len(msgs)) + ' commit messages to ' + tempPath + '/msgs.npy.')

    return msgs

def GetMsgVocab(msgs):
    '''
    Get the vocabulary of message tokens
    :param msgs - [[[tokens], 0/1], ...]
    :return: vocab - the vocabulary of message tokens. ['tk', 'tk', ...]
             maxLen - the max length of a commit message.
    '''

    # create temp folder.
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    fp = open(tempPath + 'msglen.csv', 'w')

    # get the whole tokens and the max msg length.
    tokens = []
    maxLen = 0

    # for each sample.
    for item in msgs:
        tokens.extend(item[0])
        maxLen = len(item[0]) if (len(item[0]) > maxLen) else maxLen
        fp.write(str(len(item[0])) + '\n')
    fp.close()

    # remove duplicates and get vocabulary.
    vocab = {}.fromkeys(tokens)
    vocab = list(vocab.keys())

    # print.
    print('[INFO] <GetMsgVocab> There are ' + str(len(vocab)) + ' commit message vocabulary tokens. (except \'<pad>\')')
    print('[INFO] <GetMsgVocab> The max msg length is ' + str(maxLen) + ' tokens. (hyperparameter: _MsgMaxLen_ = ' + str(_MsgMaxLen_) + ')')

    return vocab, maxLen

def GetMsgDict(vocab):
    '''
    Get the dictionary of msg vocabulary.
    :param vocab: the vocabulary of msg tokens. ['tk', 'tk', ...]
    :return: tokenDict - the dictionary of msg vocabulary.
    {'tk': 1, 'tk': 2, ..., 'tk': N, '<pad>': 0}
    '''

    # get token dict from vocabulary.
    tokenDict = {token: (index+1) for index, token in enumerate(vocab)}
    tokenDict['<pad>'] = 0

    # print.
    print('[INFO] <GetMsgDict> Create dictionary for ' + str(len(tokenDict)) + ' msg vocabulary tokens. (with \'<pad>\')')

    return tokenDict

def GetMsgEmbed(tokenDict, embedSize):
    '''
    Get the pre-trained weights for embedding layer from the dictionary of msg vocabulary.
    :param tokenDict: the dictionary of msg vocabulary.
    {'tk': 0, 'tk': 1, ..., '<pad>': N}
    :param embedSize: the dimension of the embedding vector.
    :return: preWeights - the pre-trained weights for embedding layer.
    [[n, ...], [n, ...], ...]
    '''

    # number of the vocabulary tokens.
    numTokens = len(tokenDict)

    # initialize the pre-trained weights for embedding layer.
    preWeights = np.zeros((numTokens, embedSize))
    for index in range(numTokens):
        preWeights[index] = np.random.normal(size=(embedSize,))
    print('[INFO] <GetMsgEmbed> Create pre-trained embedding weights with ' + str(len(preWeights)) + ' * ' + str(len(preWeights[0])) + ' matrix.')

    # save preWeights.
    if not os.path.exists(tempPath + '/msgPreWeights.npy'):
        np.save(tempPath + '/msgPreWeights.npy', preWeights, allow_pickle=True)
        print('[INFO] <GetMsgEmbed> Save the pre-trained weights of embedding layer to ' + tempPath + '/msgPreWeights.npy.')

    return preWeights

def GetMsgMapping(msgs, maxLen, tokenDict):
    '''
    Map the feature data into indexed data.
    :param props: the features of commit messages.
    [[[tokens], 0/1], ...]
    :param maxLen: the max length of the commit message.
    :param tokenDict: the dictionary of commit message vocabulary.
    {'tk': 1, 'tk': 2, ..., 'tk': N, '<pad>': 0}
    :return: np.array(data) - feature data.
             [[n, ...], ...]
             np.array(labels) - labels.
             [[0/1], ...]
    '''

    def PadList(dList, pad, length):
        '''
        Pad the list data to a fixed length.
        :param dList: the list data - [ , , ...]
        :param pad: the variable used to pad.
        :param length: the fixed length.
        :return: dList - padded list data. [ , , ...]
        '''

        if len(dList) <= length:
            dList.extend(pad for i in range(length - len(dList)))
        elif len(dList) > length:
            dList = dList[0:length]

        return dList

    # initialize the data and labels.
    data = []
    labels = []

    # for each sample.
    for item in msgs:
        # process tokens.
        tokens = item[0]
        tokens = PadList(tokens, '<pad>', maxLen)
        # convert tokens into numbers.
        tokens2index = []
        for tk in tokens:
            tokens2index.append(tokenDict[tk])
        data.append(tokens2index)
        # process label.
        label = item[1]
        labels.append([label])

    if _DEBUG_:
        print('[DEBUG] data:')
        print(data[0:3])
        print('[DEBUG] labels:')
        print(labels[0:3])

    # print.
    print('[INFO] <GetMsgMapping> Create ' + str(len(data)) + ' feature data with 1 * ' + str(len(data[0])) + ' vector.')
    print('[INFO] <GetMsgMapping> Create ' + str(len(labels)) + ' labels with 1 * 1 matrix.')

    # save files.
    if (not os.path.exists(tempPath + '/mdata_' + str(maxLen) + '.npy')) \
            | (not os.path.exists(tempPath + '/mlabels_' + str(maxLen) + '.npy')):
        np.save(tempPath + '/mdata_' + str(maxLen) + '.npy', data, allow_pickle=True)
        print('[INFO] <GetMsgMapping> Save the mapped numpy data to ' + tempPath + '/mdata_' + str(maxLen) + '.npy.')
        np.save(tempPath + '/mlabels_' + str(maxLen) + '.npy', labels, allow_pickle=True)
        print('[INFO] <GetMsgMapping> Save the mapped numpy labels to ' + tempPath + '/mlabels_' + str(maxLen) + '.npy.')

    return np.array(data), np.array(labels)

class MsgRNN(nn.Module):
    '''
    MsgRNN : convert a commit message into a predicted label.
    '''

    def __init__(self, preWeights, hiddenSize=32, hiddenLayers=1):
        '''
        define each layer in the network model.
        :param preWeights: tensor pre-trained weights for embedding layer.
        :param hiddenSize: node number in the hidden layer.
        :param hiddenLayers: number of hidden layer.
        '''

        super(MsgRNN, self).__init__()
        # parameters.
        class_num = 2
        vocabSize, embedDim = preWeights.size()
        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocabSize, embedding_dim=embedDim)
        self.embedding.load_state_dict({'weight': preWeights})
        self.embedding.weight.requires_grad = True
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embedDim, hidden_size=hiddenSize, num_layers=hiddenLayers, bidirectional=True)
        # Fully-Connected Layer
        self.fc = nn.Linear(hiddenSize * hiddenLayers * 2, class_num)
        # Softmax non-linearity
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        convert inputs to predictions.
        :param x: input tensor. dimension: batch_size * diff_length * 1.
        :return: self.softmax(final_out) - predictions.
        [[0.3, 0.7], [0.2, 0.8], ...]
        '''

        # x             batch_size * diff_length * 1
        embeds = self.embedding(x)
        # embeds        batch_size * diff_length * embedding_dim
        inputs = embeds.permute(1, 0, 2)
        # inputs        diff_length * batch_size * (embedding_dim + _DiffExtraDim_)
        lstm_out, (h_n, c_n) = self.lstm(inputs)
        # lstm_out      diff_length * batch_size * (hidden_size * direction_num)
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        # h_n           (num_layers * direction_num) * batch_size * hidden_size
        feature_map = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # feature_map   batch_size * (hidden_size * num_layers * direction_num)
        final_out = self.fc(feature_map)    # batch_size * class_num
        return self.softmax(final_out)      # batch_size * class_num

def MsgRNNTrain(dTrain, lTrain, dValid, lValid, preWeights, batchsize=64, learnRate=0.001, dTest=None, lTest=None):
    '''
    Train the MsgRNN model.
    :param dTrain: training data. [[n, ...], ...]
    :param lTrain: training label. [[n, ...], ...]
    :param dValid: validation data. [[n, ...], ...]
    :param lValid: validation label. [[n, ...], ...]
    :param preWeights: pre-trained weights for embedding layer.
    :param batchsize: number of samples in a batch.
    :param learnRate: learning rate.
    :param dTest: test data. [[n, ...], ...]
    :param lTest: test label. [[n, ...], ...]
    :return: model - the MsgRNN model.
    '''

    # get the mark of the test dataset.
    if dTest is None: dTest = []
    if lTest is None: lTest = []
    markTest = 1 if (len(dTest)) & (len(lTest)) else 0

    # tensor data processing.
    xTrain = torch.from_numpy(dTrain).long().cuda()
    yTrain = torch.from_numpy(lTrain).long().cuda()
    xValid = torch.from_numpy(dValid).long().cuda()
    yValid = torch.from_numpy(lValid).long().cuda()
    if (markTest):
        xTest = torch.from_numpy(dTest).long().cuda()
        yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    train = torchdata.TensorDataset(xTrain, yTrain)
    trainloader = torchdata.DataLoader(train, batch_size=batchsize, shuffle=False)
    valid = torchdata.TensorDataset(xValid, yValid)
    validloader = torchdata.DataLoader(valid, batch_size=batchsize, shuffle=False)
    if (markTest):
        test = torchdata.TensorDataset(xTest, yTest)
        testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # get training weights.
    lbTrain = [item for sublist in lTrain.tolist() for item in sublist]
    weights = []
    for lb in range(2):
        weights.append(1 - lbTrain.count(lb) / len(lbTrain))
    lbWeights = torch.FloatTensor(weights).cuda()

    # build the model of recurrent neural network.
    preWeights = torch.from_numpy(preWeights)
    model = MsgRNN(preWeights, hiddenSize=_MRnnHidSiz_, hiddenLayers=_MRnnHidLay_)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('[INFO] <MsgRNNTrain> ModelType: MsgRNN, HiddenNodes: %d, HiddenLayers: %d.' % (_MRnnHidSiz_, _MRnnHidLay_))
    print('[INFO] <MsgRNNTrain> BatchSize: %d, LearningRate: %.4f, MaxEpoch: %d, PerEpoch: %d.' % (batchsize, learnRate, _MRnnMaxEpoch_, _MRnnPerEpoch_))
    # optimizing with stochastic gradient descent.
    optimizer = optim.Adam(model.parameters(), lr=learnRate)
    # seting loss function as mean squared error.
    criterion = nn.CrossEntropyLoss(weight=lbWeights)
    # memory
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # run on each epoch.
    accList = [0]
    for epoch in range(_MRnnMaxEpoch_):
        # training phase.
        model.train()
        lossTrain = 0
        predictions = []
        labels = []
        for iter, (data, label) in enumerate(trainloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # back propagation.
            optimizer.zero_grad()  # set the gradients to zero.
            yhat = model.forward(data)  # get output
            loss = criterion(yhat, label)
            loss.backward()
            optimizer.step()
            # statistic
            lossTrain += loss.item() * len(label)
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        lossTrain /= len(dTrain)
        # train accuracy.
        accTrain = accuracy_score(labels, predictions) * 100

        # validation phase.
        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for iter, (data, label) in enumerate(validloader):
                # data conversion.
                data = data.to(device)
                label = label.contiguous().view(-1)
                label = label.to(device)
                # forward propagation.
                yhat = model.forward(data)  # get output
                # statistic
                preds = yhat.max(1)[1]
                predictions.extend(preds.int().tolist())
                labels.extend(label.int().tolist())
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        # valid accuracy.
        accValid = accuracy_score(labels, predictions) * 100
        accList.append(accValid)

        # testing phase.
        if (markTest):
            model.eval()
            predictions = []
            labels = []
            with torch.no_grad():
                for iter, (data, label) in enumerate(testloader):
                    # data conversion.
                    data = data.to(device)
                    label = label.contiguous().view(-1)
                    label = label.to(device)
                    # forward propagation.
                    yhat = model.forward(data)  # get output
                    # statistic
                    preds = yhat.max(1)[1]
                    predictions.extend(preds.int().tolist())
                    labels.extend(label.int().tolist())
                    torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            # test accuracy.
            accTest = accuracy_score(labels, predictions) * 100

        # output information.
        if (0 == (epoch + 1) % _MRnnPerEpoch_):
            strAcc = '[Epoch {:03}] loss: {:.3}, train acc: {:.3f}%, valid acc: {:.3f}%.'.format(epoch + 1, lossTrain, accTrain, accValid)
            if (markTest):
                strAcc = strAcc[:-1] + ', test acc: {:.3f}%.'.format(accTest)
            print(strAcc)
        # save the best model.
        if (accList[-1] > max(accList[0:-1])):
            torch.save(model.state_dict(), tempPath + '/model_MsgRNN.pth')
        # stop judgement.
        if (epoch >= _MRnnJudEpoch_) and (accList[-1] < min(accList[-1-_MRnnJudEpoch_:-1])):
            break

    # load best model.
    model.load_state_dict(torch.load(tempPath + '/model_MsgRNN.pth'))
    print('[INFO] <MsgRNNTrain> Finish training MsgRNN model. (Best model: ' + tempPath + '/model_MsgRNN.pth)')

    return model

def MsgRNNTest(model, dTest, lTest, batchsize=64):
    '''
    Test the MsgRNN model.
    :param model: deep learning model.
    :param dTest: test data.
    :param lTest: test label.
    :param batchsize: number of samples in a batch
    :return: predictions - predicted labels. [[0], [1], ...]
             accuracy - the total test accuracy. numeric
    '''

    # tensor data processing.
    xTest = torch.from_numpy(dTest).long().cuda()
    yTest = torch.from_numpy(lTest).long().cuda()

    # batch size processing.
    test = torchdata.TensorDataset(xTest, yTest)
    testloader = torchdata.DataLoader(test, batch_size=batchsize, shuffle=False)

    # load the model of recurrent neural network.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # testing phase.
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for iter, (data, label) in enumerate(testloader):
            # data conversion.
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            # forward propagation.
            yhat = model.forward(data)  # get output
            # statistic
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # testing accuracy.
    accuracy = accuracy_score(labels, predictions) * 100
    predictions = [[item] for item in predictions]

    return predictions, accuracy

def demoPatch():
    '''
    demo program of using both commit message and diff code to identify patches.
    '''

    # load data.
    if (not os.path.exists(tempPath + '/data.npy')): # | (not _DEBUG_)
        dataLoaded = ReadData()
    else:
        dataLoaded = np.load(tempPath + '/data.npy', allow_pickle=True)
        print('[INFO] <ReadData> Load ' + str(len(dataLoaded)) + ' raw data from ' + tempPath + '/data.npy.')

    # get the diff file properties.
    if (not os.path.exists(tempPath + '/props.npy')):
        diffProps = GetDiffProps(dataLoaded)
    else:
        diffProps = np.load(tempPath + '/props.npy', allow_pickle=True)
        print('[INFO] <GetDiffProps> Load ' + str(len(diffProps)) + ' diff property data from ' + tempPath + '/props.npy.')
    # normalize the tokens of identifiers, literals, and comments.
    diffProps = ProcessTokens(diffProps)
    diffProps = NormalizeTokens(diffProps, normType=0)

    # get the diff token vocabulary.
    diffVocab, diffMaxLen = GetDiffVocab(diffProps)
    # get the max diff length.
    diffMaxLen = _DiffMaxLen_ if (diffMaxLen > _DiffMaxLen_) else diffMaxLen
    # get the diff token dictionary.
    diffDict = GetDiffDict(diffVocab)
    # get pre-trained weights for embedding layer.
    diffPreWeights = GetDiffEmbed(diffDict, _DiffEmbedDim_)
    # get the mapping for feature data and labels.
    diffData, diffLabels = GetDiffMapping(diffProps, diffMaxLen, diffDict)
    # change the tokentypes into one-hot vector.
    diffData = UpdateTokenTypes(diffData)

    # split data into rest/test dataset.
    dataTrain, labelTrain, dataTest, labelTest = SplitData(diffData, diffLabels, 'test', rate=0.2)
    # split data into train/valid dataset.
    #dataTrain, labelTrain, dataValid, labelValid = SplitData(dataRest, labelRest, 'valid', rate=0.2)
    print('[INFO] <main> Get ' + str(len(dataTrain)) + ' TRAIN data, ' + str(len(dataTest)) + ' TEST data. (Total: ' + str(len(dataTrain)+len(dataTest)) + ')')

    # TextRNNTrain
    if (_MODEL_) & (os.path.exists(tempPath + '/model_TextRNN.pth')):
        preWeights = torch.from_numpy(diffPreWeights)
        model = MsgRNN(preWeights, hiddenSize=_TRnnHidSiz_, hiddenLayers=_TRnnHidLay_)
        model.load_state_dict(torch.load(tempPath + '/model_TextRNN.pth'))
    else:
        model = TextRNNTrain(dataTrain, labelTrain, dataTest, labelTest, preWeights=diffPreWeights,
                             batchsize=_TRnnBatchSz_, learnRate=_TRnnLearnRt_, dTest=dataTest, lTest=labelTest)

    # TextRNNTest
    predictions, accuracy = TextRNNTest(model, dataTest, labelTest, batchsize=_TRnnBatchSz_)
    _, confusion = OutputEval(predictions, labelTest, 'TextRNN')

    return

def ProcessTokens(props):
    '''
    only maintain the diff parts of the code.
    :param props: the features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    :return: props - the normalized features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    '''

    propsNew = []
    for item in props:
        # the number of tokens.
        numTokens = len(item[1])
        # item[0]: tokens, item[1]: tokenTypes, item[2]: diffTypes, item[3]: label.
        tokens = [item[0][n] for n in range(numTokens) if (item[2][n])]
        tokenTypes = [item[1][n] for n in range(numTokens) if (item[2][n])]
        diffTypes = [item[2][n] for n in range(numTokens) if (item[2][n])]
        label = item[3]
        # reconstruct sample.
        sample = [tokens, tokenTypes, diffTypes, label]
        propsNew.append(sample)
    #print(propsNew[0])

    return propsNew

def NormalizeTokens(props, normType=0):
    '''
    normalize the tokens of identifiers, literals, and comments.
    :param props: the features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    :param normType: 0 - only identify variable type and function type. VAR / FUNC
                     1 - identify the identical variable and function.  VAR0, VAR1, ... / FUNC0, FUNC1, ...
    :return: props - the normalized features of diff code.
    [[[tokens], [nums], [nums], 0/1], ...]
    '''

    for item in props:
        # get tokens and token types.
        tokens = item[0]
        tokenTypes = item[1]
        numTokens = len(tokenTypes)
        #print(tokens)
        #print(tokenTypes)
        #print(numTokens)


        # normalize literals and comments, and separate identifiers into variables and functions.
        markVar = list(np.zeros(numTokens, dtype=int))
        markFuc = list(np.zeros(numTokens, dtype=int))
        for n in range(numTokens):
            # 2: IDENTIFIER, 3: LITERAL, 5: COMMENT
            if 5 == tokenTypes[n]:
                tokens[n] = 'COMMENT'
            elif 3 == tokenTypes[n]:
                tokens[n] = 'LITERAL'
            elif 2 == tokenTypes[n]:
                # separate variable name and function name.
                if (n < numTokens-1):
                    if (tokens[n+1] == '('):
                        markFuc[n] = 1
                    else:
                        markVar[n] = 1
                else:
                    markVar[n] = 1
        #print(tokens)
        #print(markVar)
        #print(markFuc)

        # normalize variables and functions.
        if (0 == normType):
            for n in range(numTokens):
                if 1 == markVar[n]:
                    tokens[n] = 'VAR'
                elif 1 == markFuc[n]:
                    tokens[n] = 'FUNC'
        elif (1 == normType):
            # get variable dictionary.
            varList = [tokens[idx] for idx, mark in enumerate(markVar) if mark == 1]
            varVoc  = {}.fromkeys(varList)
            varVoc  = list(varVoc.keys())
            varDict = {tk: 'VAR' + str(idx) for idx, tk in enumerate(varVoc)}
            # get function dictionary.
            fucList = [tokens[idx] for idx, mark in enumerate(markFuc) if mark == 1]
            fucVoc  = {}.fromkeys(fucList)
            fucVoc  = list(fucVoc.keys())
            fucDict = {tk: 'FUNC' + str(idx) for idx, tk in enumerate(fucVoc)}
            #print(varDict)
            #print(fucDict)
            for n in range(numTokens):
                if 1 == markVar[n]:
                    tokens[n] = varDict[tokens[n]]
                elif 1 == markFuc[n]:
                    tokens[n] = fucDict[tokens[n]]
    #print(tokens)

    return props

if __name__ == '__main__':
    #demoTextRNN()
    #demoCommitMsg()
    demoPatch()
    #diffData = np.load(tempPath + '/newdata_' + str(_DiffMaxLen_) + '.npy')
    #diffLabels = np.load(tempPath + '/nlabels_' + str(_DiffMaxLen_) + '.npy')
    #dataRest, labelRest, dataTest, labelTest = SplitData(diffData, diffLabels, 'test', rate=0.2)
    #dataTrain, labelTrain, dataValid, labelValid = SplitData(dataRest, labelRest, 'valid', rate=0.2)
    #diffPreWeights = np.load(tempPath + '/preWeights.npy')
    #if (_MODEL_) & (os.path.exists(tempPath + '/model_TextRNN.pth')):
    #    preWeights = torch.from_numpy(diffPreWeights)
    #    model = TextRNN(preWeights, hiddenSize=_TRnnHidSiz_, hiddenLayers=_TRnnHidLay_)
    #    model.load_state_dict(torch.load(tempPath + '/model_TextRNN.pth'))
    #else:
    #    model = TextRNNTrain(dataTrain, labelTrain, dataValid, labelValid, preWeights=diffPreWeights,
    #                         batchsize=_TRnnBatchSz_, learnRate=_TRnnLearnRt_, dTest=dataTest, lTest=labelTest)
    #predictions, accuracy = TextRNNTest(model, dataTest, labelTest, batchsize=_TRnnBatchSz_)
    #_, confusion = OutputEval(predictions, labelTest, 'TextRNN')
