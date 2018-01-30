import torch
import torch.autograd as autograd
import numpy as np
import pickle

from utils import LabelEncoder, iobe_to_ranges


def predict_text(text, model, char_encoder: LabelEncoder, tag_encoder: LabelEncoder):
    """
    extract named entities from a string, and return entities of rasa format.
    """
    x = autograd.Variable(torch.LongTensor([char_encoder.transform(label=c, default=1) for c in text])).unsqueeze(1)
    tags_idx = model(x)
    tags = [tag_encoder.reverse(t_idx) for t_idx in np.array(tags_idx).flatten()]

    # convert tags from index to rasa format
    _ranges_of_iobe = iobe_to_ranges(tags)
    entities = []
    for _item in _ranges_of_iobe:
        start, end, entity = _item[0], _item[1] + 1, _item[2]
        entities.append({"START": start,
                         "END": end,
                         "entity": entity,
                         "value": text[start:end]
                         })
    return {"text": text, "entities": entities}


if __name__ == '__main__':
    with open("model/model_epoch_49.pkl", "rb") as f:
        model, char_encoder, tag_encoder = pickle.load(f)

    text = "设置晚上十点半的闹钟"
    print(predict_text(text, model, char_encoder, tag_encoder))
