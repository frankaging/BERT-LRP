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

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_IMDb.csv"),sep=",").values
        return self._create_examples(train_data, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "test_IMDb.csv"),sep=",").values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1] # 0: negative; 1: positive

    def _create_examples(self, lines, set_type, debug=True):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(str(line[1]))
            if i%1000==0 and debug:
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

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "test_SemEval.csv"),sep=",").values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2] # 0: negative; 1: neutral; 2: positive

    def _create_examples(self, lines, set_type, debug=True):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(str(line[1]))
            if i%1000==0 and debug:
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

    def get_train_small_examples(self, data_dir):
        """See base class."""
        train_data = sst_reader(data_dir, "stsa.fine.train")
        return self._create_examples(train_data, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = sst_reader(data_dir, "stsa.fine.test")
        return self._create_examples(test_data, "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = sst_reader(data_dir, "stsa.fine.dev")
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4] # 0: very negative ->  4: very positive

    def _create_examples(self, lines, set_type, debug=True):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(line[1])
            if i%1000==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class SST2_Processor(DataProcessor):
    """Processor for the SST data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.tsv"),sep="\t",skiprows=0).values
        return self._create_examples(train_data, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "dev.tsv"),sep="\t",skiprows=0).values
        return self._create_examples(test_data, "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "dev.tsv"),sep="\t",skiprows=0).values
        return self._create_examples(test_data, "dev")

    def get_labels(self):
        """See base class."""
        return [0, 1] # 0: very negative ->  4: very positive

    def _create_examples(self, lines, set_type, debug=True):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(line[1])
            if i%1000==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class SST3_Processor(DataProcessor):
    """Processor for the SST data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_SST.csv"),sep=",").values
        return self._create_examples(train_data, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "test_SST.csv"),sep=",").values
        return self._create_examples(test_data, "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "dev_SST.csv"),sep=",").values
        return self._create_examples(test_data, "dev")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2] # 2: non-sentiment cases

    def generate_class(self, rate):
        if rate >= 0.0 and rate <= 0.2:
            _class = 0
        elif rate >= 0.8 and rate <= 1.0:
            _class = 1
        elif rate >= 0.4 and rate <= 0.6:
            _class = 2
        else:
            _class = -1 # we exclude these cases
        return _class

    def _create_examples(self, lines, set_type, debug=True):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = self.generate_class(float(str(line[1])))
            if label == -1:
                continue
            if i%1000==0 and debug:
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

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "test_Yelp2.csv"),sep=",").values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1] # 0: negative; 1: neutral; 2: positive

    def _create_examples(self, lines, set_type, debug=True):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(str(line[1])) - 1
            if i%1000==0 and debug:
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

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_Yelp5.csv"),sep=",").values
        return self._create_examples(train_data, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "test_Yelp5.csv"),sep=",").values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4]

    def _create_examples(self, lines, set_type, debug=True):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(str(line[1])) - 1
            if i%1000==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


######################################################
#
# One example of customerized processing script
#
######################################################

def adversarial_yelp_round1_reader(src_dirname, src_filename):
    labels = {'negative': '0', 'positive': '1', 'nosentiment': '2'}
    src_filename = os.path.join(src_dirname, src_filename)
    data = []
    with open(src_filename) as f:
        for line in f:
            d = json.loads(line)
            label = labels.get(d['gold_label'])
            if label:
                data.append((d['sentence'], label))
    return data

class AdvSA_Processor(DataProcessor):
    """Processor for the AdvSA data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = adversarial_yelp_round1_reader(data_dir, "adversarial-sentiment-round01-train.jsonl")
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = adversarial_yelp_round1_reader(data_dir, "adversarial-sentiment-round01-dev.jsonl")
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = adversarial_yelp_round1_reader(data_dir, "adversarial-sentiment-round01-dev.jsonl")
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    def _create_examples(self, lines, set_type, debug=True):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(str(line[1]))
            if i%1000==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class R0Train_Processor(DataProcessor):
    """Processor for the AdvSA data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "r0train-ternary-small.tsv"),sep="\t",skiprows=0).values
        return self._create_examples(train_data, "train")

    def get_big_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "r0train-ternary-full.tsv"),sep="\t",skiprows=0).values
        return self._create_examples(train_data, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "sst-dev-ternary.tsv"),sep="\t",skiprows=0).values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2] # 0: very negative ->  4: very positive

    def _create_examples(self, lines, set_type, debug=True):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = int(line[1])
            if i%50000==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples