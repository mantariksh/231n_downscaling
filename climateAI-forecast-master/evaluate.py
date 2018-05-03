
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net as net
import model.data_loader as data_loader

def evaluate(model, loss_fn, dataloader, metrics, params):

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    eval_losses = []

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if torch.cuda.is_available():
            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)

        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        eval_losses.append(loss.data[0])
        loss_avg = torch.mean(torch.FloatTensor(eval_losses))
        logging.info("- Eval average loss : " + loss_avg)

    return loss_avg
