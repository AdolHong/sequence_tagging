import random
import json

import torch
import pickle
from tensorboardX import SummaryWriter


from model import BilstmCrf
from utils import generate_batch, prepare_ner_dataset


def do_epoch(model, dataset, batch_size, train_mode=True, is_shuffle = True):
    model.train(mode=train_mode)
    if is_shuffle:
        random.shuffle(dataset)

    acc_loss = 0

    for iter, (x, y) in enumerate(generate_batch(dataset, batch_size=batch_size)):

        # forward pass and compute loss
        loss = (- model.loss(x, y)) / x.size(1)
        acc_loss += loss.view(-1).data.tolist()[0]

        if train_mode:
            model.zero_grad()
            loss.backward()  # compute gradients
            optim.step()  # update parameters

    return acc_loss / (iter + 1)




if __name__ == '__main__':
    with open("./data.json", "r") as f:
        data = json.load(f)

    dataset, char_encoder, tag_encoder = prepare_ner_dataset(data)

    # hyperparameter
    LEARNING_RATE = 0.01

    model = BilstmCrf(char_encoder.length(), tag_encoder.label_to_idx,
                      hidden_size=200, embed_size=50, num_layers=1, dropout=0.5)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # shuffle dataset
    random.shuffle(dataset)
    train_val, val_test = int(len(dataset) * 0.9), int(len(dataset) * 0.95)
    train_set, eval_set, test_set = dataset[:train_val], dataset[train_val:val_test], dataset[val_test:]

    # google tensorboard
    writer = SummaryWriter()

    # details of train model
    num_epochs = 50
    batch_size = 10


    for epoch in range(num_epochs):
        # train
        train_loss = do_epoch(model, dataset=train_set, batch_size=batch_size, train_mode=True, is_shuffle = True)

        # evaluate
        eval_loss = do_epoch(model, dataset=eval_set, batch_size=batch_size, train_mode=False, is_shuffle=False)

        # test
        test_loss = do_epoch(model, dataset=test_set, batch_size=batch_size, train_mode=False, is_shuffle=False)

        # write loss into log, for visualization of tensorboard
        writer.add_scalars('data/loss_group', {'train': train_loss,
                                               'eval': eval_loss,
                                               'test': test_loss}, epoch)

        with open(f"model/model_epoch_{epoch}.pkl", "wb") as f:
            pickle.dump((model, char_encoder, tag_encoder), f)
