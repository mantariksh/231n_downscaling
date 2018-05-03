
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.net as crnn
import model.data_loader as data_loader
from evaluate import evaluate


def train(model, optimizer, loss_fn, dataloader):

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    losses = []

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            # move to GPU if available
            if torch.cuda.is_available():
                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)

            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute forward pass
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # save each batch loss, this can be done also once in a while
            losses.append(loss.data[0])

        # update the average loss
        loss_avg = torch.mean(torch.FloatTensor(losses))
        logging.info("- Train average loss : " + loss_avg)
        # t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
        t.update()

            return loss_avg


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, model_dir, epochs,
                       restore_file=None):

    train_losses = []
    val_losses = []

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, crnn_model, optimizer)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        loss_avg_epoch = train(model, optimizer, loss_fn, train_dataloader)
        train_losses.append(loss_avg_epoch)

        # Evaluate for one epoch on validation set
        val_loss_avg = evaluate(model, loss_fn, val_dataloader)
        val_losses.append(val_loss_avg)

        is_best = val_loss_avg <=  best_val_loss

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_ = val_loss_avg

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_loss_avg, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_loss_avg, last_json_path)



if __name__ == '__main__':

    device = "cpu"

    # training hyperparameters
    batch_size = 128
    lr = 0.0002
    epochs = 10
    data_dir = 'data'
    model_dir = 'model'


    # use GPU if available
    if torch.cuda.is_available(): device = "cuda:0"

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if device == "cuda:0": torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    # Define the model and optimizer
    crnn_model = crnn.CRNN(128).cuda() if device == "cuda:0" else crnn.CRNN(128)
    optimizer = optim.Adam(crnn_model.parameters(), lr=lr, betas=(0.5, 0.999))

    # fetch loss function and metrics
    loss_fn = crnn.loss_fn

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'val', 'test'], data_dir, batch_size)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(epochs))
    train_and_evaluate(crnn_model, train_dl, val_dl, optimizer, loss_fn, model_dir, epochs)
