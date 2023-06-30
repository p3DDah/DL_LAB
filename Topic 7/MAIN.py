import pickle

import torch
import json
import os
import pickle as pkl
import numpy as np
import sys
import logging
import itertools
from matplotlib import pyplot as plt
from datetime import datetime
import torch.cuda as cuda
#from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
max_threads = os.cpu_count()
os.environ['NUMEXPR_MAX_THREADS'] = str(max_threads)

from process import *
from dataset import *
from models import *
from plotter import *
from config import *
from utils import *
from metrics import *
from optimizer import *
from regularizers import *


# TODO: filters might be wrong initialized
# TODO: adjust arrays so that we have dynamic lrs, batches, optimizers
# TODO: plot with different learning rates
# TODO: more regularizer -> "N3", "F2"
# TODO: another file for savings

# TODO: try?
#  different Embedding sizes
#  different optimizer
#  different init sizes
#  different regularizer values
#  different learning rates, dynamic learning rates
#  different margin

# TODO: Implement?
#  ->
def train(args):
    cuda.empty_cache()

    with cuda.device(0):
        allocated = cuda.memory_allocated()
        reserved = cuda.memory_reserved()

    class instants:
        # models
        NaiveTransE = NaiveTransE
        VectorTransE = VectorTransE
        # regularizers
        F2 = F2
        N3 = N3

    save_dir = get_savedir(args.model, args.dataset, args.num_save_files)

    # file logger
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.INFO,
        filename=os.path.join(save_dir, "train.log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}\n".format(save_dir))

    if "cuda" in args.device:
        logging.info("CUDA available: {}".format(torch.cuda.is_available()))
        if torch.cuda.is_available():
            logging.info("Using cuda device id: {}".format(torch.cuda.current_device()))
            logging.info("Name of cuda device: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
            logging.info("Calculating on GPU (cuda)\n")
        else:
            args.device = "cpu"
            logging.info("Setting device to CPU\n")
    else:
        logging.info("Calculating on CPU\n")

    # create pickle files
    dataset_path = os.path.join("datasets")
    if not os.path.exists(os.path.join(dataset_path, args.dataset, "test.txt.pickle")):
        ds_read(dataset_path)

    # load data
    dataset_path = os.path.join("datasets", args.dataset)
    logging.info("Loading the datasets and looking for filter dataset")
    train = TKGEDataset(dataset_path, 'train.txt.pickle', args.debug)
    valid = TKGEDataset(dataset_path, 'valid.txt.pickle', args.debug)
    test = TKGEDataset(dataset_path, 'test.txt.pickle', args.debug)
    args.sizes = train.get_shape()

    x = train[:]
    x_v = valid[:]
    x_t = test[:]

    if os.path.exists(os.path.join("savings", "all_quadruple_rank_t.pth")) and \
        os.path.exists(os.path.join("savings", "all_quadruple_rank_h.pth")) and \
            os.path.exists(os.path.join("savings", "all_quadruple_rank_raw_t.pth")) and \
            os.path.exists(os.path.join("savings", "all_quadruple_rank_raw_h.pth")):
            all_quadruple_rank_t = torch.load("savings/all_quadruple_rank_t.pth")
            all_quadruple_rank_h = torch.load("savings/all_quadruple_rank_h.pth")
            all_quadruple_rank_raw_t = torch.load("savings/all_quadruple_rank_raw_t.pth")
            all_quadruple_rank_raw_h = torch.load("savings/all_quadruple_rank_raw_h.pth")
            logging.info("Filter files found and loaded")
    else:
        all_quadruple_rank_t, all_quadruple_rank_h, all_quadruple_rank_raw_t, all_quadruple_rank_raw_h = \
            train.merge_and_create_dic(x, x_v, x_t)
        torch.save(all_quadruple_rank_t, os.path.join("savings", "all_quadruple_rank_t.pth"))
        torch.save(all_quadruple_rank_h, os.path.join("savings", "all_quadruple_rank_h.pth"))
        torch.save(all_quadruple_rank_raw_t, os.path.join("savings", "all_quadruple_rank_raw_t.pth"))
        torch.save(all_quadruple_rank_raw_h, os.path.join("savings", "all_quadruple_rank_raw_h.pth"))
        logging.info("Filter files created and loaded")

    x = x.t()
    x_v = x_v.t()
    x_t = x_t.t()

    logging.info("Training Dataset Shape: " + "\t\t" + str(train.get_shape()))
    logging.info("Validation Dataset Shape: " + "\t\t" + str(valid.get_shape()))
    logging.info("Test Dataset Shape: " + "\t\t\t" + str(test.get_shape()) + "\n")

    logging.info("Initializing the model")

    # save config
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    # create model
    model = getattr(instants, args.model)(args)
    total = count_params(model)
    logging.info("Total number of trainable parameters {}\n\n".format(total))
    model.to(device)

    # get optimizer
    regularizer = getattr(instants, args.regularizer)(args.reg)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = TKGE_Optimizer(model, regularizer, optim_method, args.batch_size, args.neg_sample_size, args.device, args)

    logging.info("Everything is initialized!")
    logging.info(f"Processing pipeline:\t\t Training->{args.train}, Evaluation->{args.eval}, Testing->{args.test}")
    logging.info(f"Plotting:\t\t\t\t\t Creating->{args.plt}, Saving->{args.plt_save}, Showing->{args.plt_show}")
    if args.plt:
        logging.info(f"Plot types:\t\t\t\t Loss->{args.plt}, MRR->{args.plt_mrr}, Hits10->{args.plt_hits10}"
                     f", Hits3->{args.plt_hits3}, Hits1->{args.plt_hits1}")
    logging.info(f"Parameters:\n"
                 f"\t\t Dataset: {args.dataset}\n \t\t Model: {args.model}\n \t\t Regularizer: {args.regularizer}\n"
                 f"\t\t Regularization weight: {args.reg}\n \t\t Optimizer: {args.optimizer}\n \t\t Epochs: {args.max_epochs}\n"
                 f"\t\t Early stopping after: {args.patience} epochs\n \t\t Validation step size: {args.valid}\n"
                 f"\t\t Embedding dimension: {args.rank}\n \t\t Batch size: {args.batch_size}\n \t\t Negative sample size: {args.neg_sample_size}\n"
                 f"\t\t Initial embeddings' scale: {args.init_size}\n \t\t Learning rate: {args.learning_rate}\n"
                 f"\t\t Bias type: {args.bias}\n \t\t Margin: {args.margin}\n \t\t Debug mode: {args.debug}\n")
    cont = input("Continue process? [y/n]\n") if not args.debug else "y"
    if cont != "y": return
    print()

    cuda.empty_cache()
    if args.train:
        logging.info("Start training!\n")
        early_stopping_counter = 0
        best_mrr = None
        best_epoch = None
        train_total_loss, train_mean_loss = [], []
        mr, mrr, hits10, hits3, hits1 = [], [], [], [], []
        mr_raw, mrr_raw, hits10_raw, hits3_raw, hits1_raw = [], [], [], [], []

        for step in range(args.max_epochs):
            # Train step
            model.train()
            total_loss, mean_loss = optimizer.epoch(x, step, args.max_epochs)
            logging.info(f"Epoch {step+1}/{args.max_epochs} | average train loss: {mean_loss:.4f}, total loss: {total_loss:.4f}\n")
            train_total_loss.append(total_loss)
            train_mean_loss.append(mean_loss)

            if (step + 1) % args.valid == 0 and args.eval:
                # Valid step
                model.eval()
                valid_total_loss, valid_mean_loss = optimizer.calculate_valid_loss(x_v)
                logging.info(f"Epoch {step+1}/{args.max_epochs} | average evaluation loss: {valid_mean_loss:.4f}, "
                             f"total evaluation loss: {valid_total_loss:.4f}\n")

                # get metrics
                metrics = ranking(x_v, model, all_quadruple_rank_t, all_quadruple_rank_h, all_quadruple_rank_raw_t, all_quadruple_rank_raw_h, args.valid_batch)
                logging.info(format_metrics(metrics, split="valid"))

                mr.append(metrics["MR"])
                mrr.append(metrics["MRR"])
                hits10.append(metrics["HITS10"])
                hits3.append(metrics["HITS3"])
                hits1.append(metrics["HITS1"])

                mr_raw.append(metrics["MR_RAW"])
                mrr_raw.append(metrics["MRR_RAW"])
                hits10_raw.append(metrics["HITS10"])
                hits3_raw.append(metrics["HITS3"])
                hits1_raw.append(metrics["HITS1"])

                valid_mrr = metrics["MRR"]
                if not best_mrr or valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    early_stopping_counter = 0
                    best_epoch = step + 1
                    logging.info("\t Saving model at epoch {} in {}\n".format(step + 1, save_dir))
                    torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                    model.to(device)
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter == args.patience:
                        logging.info("\t Early stopping")
                        break
                    elif early_stopping_counter == args.patience // 2 and args.dyn_lr:
                        logging.info("\t Reducing learning rate")
                        optimizer.reduce_lr()
            print("")
        logging.info("\t Optimization finished")
        if not best_mrr:
            torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
        else:
            logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
            model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
        model.to(device)
        model.eval()

        if args.eval:
            torch.save(model.state_dict(), os.path.join(save_dir, args.model + '.ckpt'))
            with open(os.path.join(save_dir, args.model) + "_total_loss", "wb") as fp:
                pkl.dump(train_total_loss, fp)
            with open(os.path.join(save_dir, args.model) + "_mean_loss", "wb") as fp:
                pkl.dump(train_mean_loss, fp)

            with open(os.path.join(save_dir, args.model) + "_mr", "wb") as fp:
                pkl.dump(mr, fp)
            with open(os.path.join(save_dir, args.model) + "_mrr", "wb") as fp:
                pkl.dump(mrr, fp)
            with open(os.path.join(save_dir, args.model) + "_hits10", "wb") as fp:
                pkl.dump(hits10, fp)
            with open(os.path.join(save_dir, args.model) + "_hits3", "wb") as fp:
                pkl.dump(hits3, fp)
            with open(os.path.join(save_dir, args.model) + "_hits1", "wb") as fp:
                pkl.dump(hits1, fp)

            with open(os.path.join(save_dir, args.model) + "_mr_raw", "wb") as fp:
                pkl.dump(mr_raw, fp)
            with open(os.path.join(save_dir, args.model) + "_mrr_raw", "wb") as fp:
                pkl.dump(mrr_raw, fp)
            with open(os.path.join(save_dir, args.model) + "_hits10_raw", "wb") as fp:
                pkl.dump(hits10_raw, fp)
            with open(os.path.join(save_dir, args.model) + "_hits3_raw", "wb") as fp:
                pkl.dump(hits3_raw, fp)
            with open(os.path.join(save_dir, args.model) + "_hits1_raw", "wb") as fp:
                pkl.dump(hits1_raw, fp)

    # Validation metrics
    if args.eval:
        metrics = ranking(x_v, model, all_quadruple_rank_t, all_quadruple_rank_h, all_quadruple_rank_raw_t, all_quadruple_rank_raw_h, args.valid_batch)
        logging.info(format_metrics(metrics, split="valid "))

    # Test metrics
    if args.test:
        metrics = ranking(x_v, model, all_quadruple_rank_t, all_quadruple_rank_h, all_quadruple_rank_raw_t, all_quadruple_rank_raw_h, args.valid_batch)
        logging.info(format_metrics(metrics, split="test "))

    if args.plt:
        logging.info("Select the folder, which data you want to plot")
        if not(args.train and args.save_train and args.eval): save_dir = select_directory_dialog()
        if save_dir:
            print("Ausgewählter Ordnerpfad:", save_dir)
        else:
            print("Kein Ordner ausgewählt.")

        model.load_state_dict(torch.load(os.path.join(save_dir, args.model) + '.ckpt'))
        with open(os.path.join(save_dir, args.model) + "_total_loss", "rb") as fp:
            train_total_loss = pkl.load(fp)
        with open(os.path.join(save_dir, args.model) + "_mean_loss", "rb") as fp:
            train_mean_loss = pkl.load(fp)

        with open(os.path.join(save_dir, args.model) + "_mr", "rb") as fp:
            mr = pkl.load(fp)
        with open(os.path.join(save_dir, args.model) + "_mrr", "rb") as fp:
            mrr = pkl.load(fp)
        with open(os.path.join(save_dir, args.model) + "_hits10", "rb") as fp:
            hits10 = pkl.load(fp)
        with open(os.path.join(save_dir, args.model) + "_hits3", "rb") as fp:
            hits3 = pkl.load(fp)
        with open(os.path.join(save_dir, args.model) + "_hits1", "rb") as fp:
            hits1 = pkl.load(fp)

        with open(os.path.join(save_dir, args.model) + "_mr_raw", "rb") as fp:
            mr_raw = pkl.load(fp)
        with open(os.path.join(save_dir, args.model) + "_mrr_raw", "rb") as fp:
            mrr_raw = pkl.load(fp)
        with open(os.path.join(save_dir, args.model) + "_hits10_raw", "rb") as fp:
            hits10_raw = pkl.load(fp)
        with open(os.path.join(save_dir, args.model) + "_hits3_raw", "rb") as fp:
            hits3_raw = pkl.load(fp)
        with open(os.path.join(save_dir, args.model) + "_hits1_raw", "rb") as fp:
            hits1_raw = pkl.load(fp)

        plotter(torch.tensor(train_total_loss), "Total-Loss", args.model, args.plt_save,
                args.plt_show, save_dir) if \
            args.plt_loss else None
        plotter(torch.tensor(train_mean_loss), "Mean-Loss", args.model, args.plt_save,
                args.plt_show, save_dir) if \
            args.plt_loss else None
        plotter(torch.tensor(mrr), "MR", args.model, args.plt_save, args.plt_show, save_dir) if \
            args.plt_mr else None
        plotter(torch.tensor(mrr), "MRR", args.model, args.plt_save, args.plt_show, save_dir) if \
            args.plt_mrr else None
        plotter(torch.tensor(hits10), "Hits10", args.model, args.plt_save, args.plt_show, save_dir) if \
            args.plt_hits10 else None
        plotter(torch.tensor(hits3), "Hits3", args.model, args.plt_save, args.plt_show, save_dir) if \
            args.plt_hits3 else None
        plotter(torch.tensor(hits1), "Hits1", args.model, args.plt_save, args.plt_show, save_dir) if \
            args.plt_hits1 else None

        plotter(torch.tensor(mr_raw), "MR_RAW", args.model, args.plt_save, args.plt_show, save_dir) if \
            args.plt_mr else None
        plotter(torch.tensor(mrr_raw), "MRR_Raw", args.model, args.plt_save, args.plt_show, save_dir) if \
            args.plt_mrr else None
        plotter(torch.tensor(hits10_raw), "Hits10_Raw", args.model, args.plt_save, args.plt_show, save_dir) if \
            args.plt_hits10 else None
        plotter(torch.tensor(hits3_raw), "Hits3_Raw", args.model, args.plt_save, args.plt_show, save_dir) if \
            args.plt_hits3 else None
        plotter(torch.tensor(hits1_raw), "Hits1_Raw", args.model, args.plt_save, args.plt_show, save_dir) if \
            args.plt_hits1 else None

    #os.system("shutdown.exe /h")

def select_directory_dialog():
    app = QApplication([])
    window = QMainWindow()

    directory_path = QFileDialog.getExistingDirectory(window, "Ordner auswählen")

    app.quit()

    return directory_path

if __name__ == "__main__":
    models = ["NaiveTransE", "VectorTransE"]
    num_lr = [0.01] # [0.001, 0.0001] #TODO: only choose one learning rate
    opt = ["Adagrad", "Adam"]
    num_reg = [0, 0.1, 0.01]
    num_neg_samp = [0, 100]
    ds = ["ICEWS05-15", "ICEWS14"]
    rank = [10, 50, 100] # rank 100 first
    batch_size = [1000]

    parameter_combinations = list(
        itertools.product(models, num_lr, opt, num_reg, num_neg_samp,
                          ds, rank, batch_size))
    print(len(parameter_combinations)*2)

    args = configurations()
    train(args)

    exit()

    #loss = ["loglikelihoodloss", "marginrankingloss"] # separat nochmal starten



    for combination in parameter_combinations:
        args.model, args.learning_rate, args.optimizer, args.reg, args.neg_sample_size, \
            args.init_size, args.dataset, args.rank, args.batch_size, args.regularizer = combination
        train(args)