from functools import reduce

import torch
import torch.autograd as autograd

from model import PAD, PAD_IDX, UNK, UNK_IDX


class LabelEncoder:
    def __init__(self, init_dict={}):
        self.label_to_idx = init_dict.copy()
        self.idx_to_label = {
            self.label_to_idx[key]: key for key in self.label_to_idx}

    def length(self):
        return len(self.label_to_idx.keys())

    def add(self, label):
        if label not in self.label_to_idx.keys():
            self.idx_to_label[len(self.label_to_idx)] = label
            self.label_to_idx[label] = len(self.label_to_idx)

    def add_transform(self, label):
        self.add(label)
        return self.label_to_idx[label]

    def reverse(self, idx, default=None):
        return self.idx_to_label.get(idx, default)

    def transform(self, label, default=None):
        return self.label_to_idx.get(label, default)

    def extract(self):
        return self.label_to_idx, self.idx_to_label





def iobe(label, length):
    """
    e.g :   iobe("LOC", 2)  =  ['B-LOC', 'E-LOC']
    """
    assert length > 0
    targets = ["B" + "-" + label]
    if length > 1:
        for idx in range(1, length - 1):
            targets.append("I" + "-" + label)
        targets.append("E" + "-" + label)
    return targets


def iobe_to_ranges(tags):
    """
    IOBE -> Ranges
    reference to:  https://github.com/glample/tagger/blob/master/utils.py
    """
    ranges = []
    state = {"begin": None, "type": None}

    def check_if_closing_range():
        if i == len(tags) - 1 or tags[i + 1].split('-')[0] == 'O' or \
                tags[i + 1].split('-')[0] == 'B' or \
                tags[i].split('-')[0] == "E":
            ranges.append((state["begin"], i, state["type"]))
            state["begin"], state["type"] = None, None

    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'O':
            pass
        elif tag.split('-')[0] == 'B':
            state["begin"] = i
            state["type"] = tag.split('-')[1]
            check_if_closing_range()
        elif tag.split('-')[1] != state["type"]:  # 避免出现 B-LOC, I-PER, E_LOC 的情况
            state["begin"], state["type"] = None, None
        elif tag.split('-')[0] == 'E':
            check_if_closing_range()
    return ranges


def rasa_to_vec(raw_text: str, entities: list, char_encoder, tag_encoder, tag_func=iobe):
    """
    convert  example from rasa format to vector format.
    """
    x = list(raw_text)
    y = ["O"] * len(raw_text)
    for e in entities:
        e_value, e_label, e_start = e["value"], e['entity'], e['start']
        tags = tag_func(label=e_label, length=len(e_value))
        for _i, _t in enumerate(tags):
            y[e_start + _i] = _t

    char_idx = [char_encoder.transform(
        _char, default=UNK_IDX) for _char in raw_text]
    tag_idx = [tag_encoder.transform(_tag, default=UNK_IDX) for _tag in y]

    return x, y, char_idx, tag_idx


def seq_pad(seq: list, pad_item, pad_len):
    """
    expand sequence to specific length.
    """
    seq = seq + [pad_item] * (pad_len - len(seq))
    return seq


def batch_pad(batch_data: list):
    """
    padding a batch of sequence to uniform length.
    """
    max_len = len(batch_data[0][0])
    for idx, (x, y) in enumerate(batch_data):
        x = seq_pad(x, pad_item=PAD_IDX, pad_len=max_len)
        y = seq_pad(y, pad_item=PAD_IDX, pad_len=max_len)
        batch_data[idx] = (x, y)
    return batch_data


def generate_batch(data: list, batch_size, batch_first=False):
    """
    generator to generate batch for training.    
    """
    idx = 0
    while idx < len(data):
        batch_data = data[idx:(idx + batch_size)]
        idx += batch_size

        batch_data = sorted(batch_data, key=lambda x: len(x[0]), reverse=True)
        batch_data = batch_pad(batch_data)

        x = autograd.Variable(torch.LongTensor(
            list(map(lambda item: item[0], batch_data))))
        y = autograd.Variable(torch.LongTensor(
            list(map(lambda item: item[1], batch_data))))

        if not batch_first:
            x, y = x.transpose(0, 1), y.transpose(0, 1)

        yield x, y




def prepare_encoder(examples: list):
    char_encoder = LabelEncoder({PAD: PAD_IDX, UNK: UNK_IDX})
    _chars = set(reduce(lambda x, y: x + y,
                        map(lambda x: x["text"], examples)))
    for c in _chars:
        char_encoder.add(c)

    # iobe
    tag_encoder = LabelEncoder({PAD: PAD_IDX, "O": 1})
    _entities = reduce(lambda x, y: x + y,
                       map(lambda x: x["entities"], examples))
    _labels = set(map(lambda x: x["entity"], _entities))
    for _label in _labels:
        tag_encoder.add("B-" + _label)
        tag_encoder.add("I-" + _label)
        tag_encoder.add("E-" + _label)
    return char_encoder, tag_encoder


def prepare_ner_dataset(rasa_data: dict):
    examples = rasa_data["rasa_nlu_data"]["common_examples"]
    char_encoder, tag_encoder = prepare_encoder(examples)

    training_data = []
    for e in examples:
        _, _, x_idx, y_idx = rasa_to_vec(
            e["text"], e["entities"], char_encoder, tag_encoder)
        training_data.append((x_idx, y_idx))

    return training_data, char_encoder, tag_encoder
