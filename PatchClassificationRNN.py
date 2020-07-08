'''
    PatchClassificationRNN
'''

import os
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize
import tokenize
import clang.cindex
import clang.enumerations
import csv

rootPath = './'
dataPath = rootPath + '/data/'
sDatPath = dataPath + '/security_patch/'
pDatPath = dataPath + '/positives/'
nDatPath = dataPath + '/negatives/'
tempPath = rootPath + '/temp/'

_DEBUG_ = 1


class Tokenizer:
    # creates the object, does the inital parse
    def __init__(self, path):
        self.index = clang.cindex.Index.create()
        self.tu = self.index.parse(path)
        self.path = self.extract_path(path)

    # To output for split_functions, must have same path up to last two folders
    def extract_path(self, path):
        return "".join(path.split("/")[:-2])

    # does futher processing on a literal token
    def process_literal(self, literal):
        cursor_kind = clang.cindex.CursorKind
        kind = literal.cursor.kind

        if kind == cursor_kind.INTEGER_LITERAL:
            return ["NUM"]

        if kind == cursor_kind.FLOATING_LITERAL:
            return ["NUM"]

        if kind == cursor_kind.IMAGINARY_LITERAL:
            return ["NUM"]

        if kind == cursor_kind.STRING_LITERAL:
            return ["STRING"]

        if kind == cursor_kind.CHARACTER_LITERAL:
            return ["CHAR"]

        if kind == cursor_kind.CXX_BOOL_LITERAL_EXPR:
            return ["BOOL"]

        # catch all other literals
        return ["LITERAL"]

    # filters out unwanted punctuation
    def process_puntuation(self, punctuation):
        spelling = punctuation.spelling

        # ignore certain characters
        if spelling in ["{", "}", "(", ")", ";"]:
            return None

        return [spelling]

    # further processes and identifier token
    def process_ident(self, ident):
        # are we a "special" ident?
        if ident.spelling in ["std", "cout", "cin", "vector", "pair", "string", "NULL", "size_t"]:
            return [ident.spelling]

        # are we a declaration?
        if ident.cursor.kind.is_declaration():
            return ["DEC"]

        # are we a reference kind?
        if ident.cursor.kind.is_reference():
            return ["REF"]

        # are we a variable use?
        if ident.cursor.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
            return ["USE"]

        # catch all others
        return ["IDENT"]

    # tokenizes the contents of a specific cursor
    def full_tokenize_cursor(self, cursor):
        tokens = cursor.get_tokens()

        # return final tokens as a list
        result = []

        for token in tokens:
            if token.kind.name == "COMMENT":
                # ignore all comments
                continue

            if token.kind.name == "PUNCTUATION":
                punct_or_none = self.process_puntuation(token)

                # add only if not ignored
                if punct_or_none != None:
                    result += punct_or_none

                continue

            if token.kind.name == "LITERAL":
                result += self.process_literal(token)
                continue

            if token.kind.name == "IDENTIFIER":
                result += self.process_ident(token)
                continue

            if token.kind.name == "KEYWORD":
                result += [token.spelling]

        return result

    # tokenizes the entire document
    def full_tokenize(self):
        cursor = self.tu.cursor
        return self.full_tokenize_cursor(cursor)

    # returns a list of function name / function / filename tuples
    def split_functions(self, method_only):
        results = []
        cursor_kind = clang.cindex.CursorKind

        # query all children for methods, and then tokenize each
        cursor = self.tu.cursor
        for c in cursor.get_children():
            filename = c.location.file.name if c.location.file != None else "NONE"
            extracted_path = self.extract_path(filename)

            if (c.kind == cursor_kind.CXX_METHOD or (
                    method_only == False and c.kind == cursor_kind.FUNCTION_DECL)) and extracted_path == self.path:
                name = c.spelling
                tokens = self.full_tokenize_cursor(c)
                filename = filename.split("/")[-1]
                results += [(name, tokens, filename)]

        return results


def main():
    # load data.
    if not os.path.exists(tempPath + '/data.npy') | (not _DEBUG_):
        dataLoaded = ReadData()
    else:
        dataLoaded = np.load(tempPath + '/data.npy', allow_pickle=True)

    CreateDiffVocab(dataLoaded)
    # get vocab, dict, preweights, mapping

    return

def ReadData():

    def ReadCommitMsg(filename):
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

    # initialize data.
    dataLoaded = []

    # read security patch data.
    for root, ds, fs in os.walk(sDatPath):
        for file in fs:
            filename = os.path.join(root, file).replace('\\', '/')
            commitMsg = ReadCommitMsg(filename)
            diffLines = ReadDiffLines(filename)
            dataLoaded.append([commitMsg, diffLines, 1])

    # read positive data.
    for root, ds, fs in os.walk(pDatPath):
        for file in fs:
            filename = os.path.join(root, file).replace('\\', '/')
            commitMsg = ReadCommitMsg(filename)
            diffLines = ReadDiffLines(filename)
            dataLoaded.append([commitMsg, diffLines, 1])

    # read negative data.
    for root, ds, fs in os.walk(nDatPath):
        for file in fs:
            filename = os.path.join(root, file).replace('\\', '/')
            commitMsg = ReadCommitMsg(filename)
            diffLines = ReadDiffLines(filename)
            dataLoaded.append([commitMsg, diffLines, 0])

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
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    if not os.path.exists(tempPath + '/data.npy'):
        np.save(tempPath + '/data.npy', dataLoaded, allow_pickle=True)

    return dataLoaded

def CreateDiffVocab(data):
    def RemoveSign(line):
        return ' ' + line[1:] if (line[0] == '+') or (line[0] == '-') else line

    def GetString(lines):
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

    def GetClangTokens(line):
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
        tknzr = TweetTokenizer()
        tokens = tknzr.tokenize(RemoveSign(line))
        return tokens

    #lines = data[0][1]
    #print(lines)
    #hunk = data[0][1][0]
    #print(hunk)
    #line = data[0][1][0][0]
    #print(line)

    # for each sample data[n].
    numData = len(data)
    for n in range(numData):
        diffLines = data[n][1]


    lines = data[0][1]
    for hunk in lines:
        for line in hunk:
            print(line, end='')
            #tokens = GetWordTokens(line)
            #print(tokens)
            tokens, tokenTypes, diffTypes = GetClangTokens(line)
            print(tokens)
            print(tokenTypes)
            print(diffTypes)
            print('-----------------------------------------------------------------------')




    return

if __name__ == '__main__':
    main()