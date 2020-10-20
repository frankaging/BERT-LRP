# coding=utf-8

"""Processors for different tasks."""

import csv
import os

import pandas as pd
import pickle

from util.tokenization import *


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

class SST5_Processor(DataProcessor):
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
        return [0, 1, 2, 3, 4] # 0: very negative ->  4: very positive

    def generate_class(self, rate):
        if rate >= 0.0 and rate <= 0.2:
            _class = 0
        elif rate > 0.2 and rate <= 0.4:
            _class = 1
        elif rate > 0.4 and rate <= 0.6:
            _class = 2
        elif rate > 0.6 and rate <= 0.8:
            _class = 3
        elif rate > 0.8 and rate <= 1.0:
            _class = 4
        else:
            assert(False)
        return _class

    def _create_examples(self, lines, set_type, debug=True):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = None
            label = self.generate_class(float(str(line[1])))
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
        return [0, 1] # 0: very negative ->  4: very positive

    def generate_class(self, rate):
        if rate >= 0.0 and rate <= 0.4:
            _class = 0
        elif rate > 0.6 and rate <= 1.0:
            _class = 1
        else:
            _class = -1
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