# coding=utf-8

"""Processors for different tasks."""

import csv
import os
import json

import pandas as pd
import pickle

from util.tokenization import *

import re
import sys

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
        
class IMDb_Processor(DataProcessor):
    """Processor for the IMBb data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir, sentence_limit=None):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_IMDb.csv"),sep=",").values
        return self._create_examples(train_data, "train", sentence_limit=sentence_limit)

    def get_test_examples(self, data_dir, sentence_limit=12500):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "test_IMDb.csv"),sep=",").values
        return self._create_examples(test_data, "test", sentence_limit=sentence_limit)

    def get_labels(self):
        """See base class."""
        return [0, 1] # 0: negative; 1: positive

    def _create_examples(self, lines, set_type, debug=True, sentence_limit=None):
        examples = []
        print("sentence limit=",sentence_limit)
        for (i, line) in enumerate(lines):
            if sentence_limit:
                if i > sentence_limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(str(line[1]))
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class SemEval_Processor(DataProcessor):
    """Processor for the SemEval data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_SemEval.csv"),sep=",").values
        return self._create_examples(train_data, "train")

    def get_test_examples(self, data_dir, sentence_limit=None):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "test_SemEval.csv"),sep=",").values
        return self._create_examples(test_data, "test", sentence_limit=sentence_limit)

    def get_labels(self):
        """See base class."""
        return [0, 1, 2] # 0: negative; 1: neutral; 2: positive

    def _create_examples(self, lines, set_type, debug=True, sentence_limit=None):
        examples = []
        print("sentence limit=",sentence_limit)
        for (i, line) in enumerate(lines):
            if sentence_limit:
                if i > sentence_limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(str(line[1]))
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"\. \. \.", "\.", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def sst_reader(src_dirname, src_filename):
    src_filename = os.path.join(src_dirname, src_filename)
    data = []
    with open(src_filename) as f:
        for line in f:
            div = line.index(' ')
            sentence = clean_str(line[div+1:])
            label = line[:div]
            if label:
                data.append((sentence, label))
    return data

class SST5_Processor(DataProcessor):
    """Processor for the SST data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = sst_reader(data_dir, "stsa.fine.train")
        return self._create_examples(train_data, "train")

    def get_test_examples(self, data_dir, sentence_limit=None):
        """See base class."""
        test_data = sst_reader(data_dir, "stsa.fine.test")
        return self._create_examples(test_data, "test", sentence_limit=sentence_limit)

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = sst_reader(data_dir, "stsa.fine.dev")
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4] # 0: very negative ->  4: very positive

    def _create_examples(self, lines, set_type, debug=True, sentence_limit=None):
        examples = []
        print("sentence limit=",sentence_limit)
        for (i, line) in enumerate(lines):
            if sentence_limit:
                if i > sentence_limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(line[1])
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class SST3_Processor(DataProcessor):
    """Processor for the SST3 data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "sst-tenary-train.tsv"),delimiter="\t").values
        return self._create_examples(train_data, "train")

    def get_test_examples(self, data_dir, sentence_limit=None):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "sst-tenary-dev.tsv"),delimiter="\t").values
        return self._create_examples(test_data, "test", sentence_limit=sentence_limit)

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    def _create_examples(self, lines, set_type, debug=True, sentence_limit=None):
        examples = []
        print("sentence limit=",sentence_limit)
        for (i, line) in enumerate(lines):
            if sentence_limit:
                if i > sentence_limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(str(line[1]))
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class QNLI_Processor(DataProcessor):
    """Processor for the QNLI data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.tsv"),delimiter="\t").values
        return self._create_examples(train_data, "train")

    def get_test_examples(self, data_dir, sentence_limit=None):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "dev.tsv"),delimiter="\t").values
        return self._create_examples(test_data, "test", sentence_limit=sentence_limit)

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type, debug=True, sentence_limit=None):
        examples = []
        print("sentence limit=",sentence_limit)
        for (i, line) in enumerate(lines):
            if sentence_limit:
                if i > sentence_limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = convert_to_unicode(str(line[1]))
            label = int(str(line[2]))
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
class MRPC_Processor(DataProcessor):
    """Processor for the MRPC data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.tsv"),delimiter="\t").values
        return self._create_examples(train_data, "train")

    def get_test_examples(self, data_dir, sentence_limit=None):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "dev.tsv"),delimiter="\t").values
        return self._create_examples(test_data, "test", sentence_limit=sentence_limit)

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type, debug=True, sentence_limit=None):
        examples = []
        print("sentence limit=",sentence_limit)
        for (i, line) in enumerate(lines):
            if sentence_limit:
                if i > sentence_limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = convert_to_unicode(str(line[1]))
            label = int(str(line[2]))
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
class Yelp2_Processor(DataProcessor):
    """Processor for the Yelp2 data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_Yelp2.csv"),sep=",").values
        return self._create_examples(train_data, "train")

    def get_test_examples(self, data_dir, sentence_limit=None):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "test_Yelp2.csv"),sep=",").values
        return self._create_examples(test_data, "test", sentence_limit=sentence_limit)

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type, debug=True, sentence_limit=None):
        examples = []
        print("sentence limit=",sentence_limit)
        for (i, line) in enumerate(lines):
            if sentence_limit:
                if i > sentence_limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(str(line[1])) - 1
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class Yelp5_Processor(DataProcessor):
    """Processor for the Yelp5 data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir, sentence_limit=25000):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_Yelp5.csv"),sep=",").values
        return self._create_examples(train_data, "train", sentence_limit=sentence_limit)

    def get_test_examples(self, data_dir, sentence_limit=12500):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "test_Yelp5.csv"),sep=",").values
        return self._create_examples(test_data, "test", sentence_limit=sentence_limit)

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4]

    def _create_examples(self, lines, set_type, debug=True, sentence_limit=None):
        examples = []
        print("sentence limit=",sentence_limit)
        for (i, line) in enumerate(lines):
            if sentence_limit:
                if i > sentence_limit:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(str(line[1])) - 1
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples