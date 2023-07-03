import torch
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def configurations():
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Embedding"
    )

    parser.add_argument(
        "--train", default=True, type=bool, choices=[True, False], help="training of the neural net"
    )
    parser.add_argument(
        "--eval", default=True, type=bool, choices=[True, False], help="evaluation of neural net (during training "
                                                                       "and at the end)"
    )
    parser.add_argument(
        "--test", default=True, type=bool, choices=[True, False], help="test of the neural net"
    )
    parser.add_argument(
        "--save_train", default=True, type=bool, choices=[True, False], help="save parameters of neural net"
    )
    parser.add_argument(
        "--plt", default=True, type=bool, choices=[True, False], help="create plots"
    )
    parser.add_argument(
        "--plt_save", default=True, type=bool, choices=[True, False], help="save plots"
    )
    parser.add_argument(
        "--plt_show", default=False, type=bool, choices=[True, False], help="show plots"
    )
    parser.add_argument(
        "--plt_loss", default=True, type=bool, choices=[True, False], help="create loss plots"
    )
    parser.add_argument(
        "--plt_mr", default=True, type=bool, choices=[True, False], help="create loss plots"
    )
    parser.add_argument(
        "--plt_mrr", default=True, type=bool, choices=[True, False], help="create mrr plots"
    )
    parser.add_argument(
        "--plt_hits10", default=True, type=bool, choices=[True, False], help="create hits10 plots"
    )
    parser.add_argument(
        "--plt_hits3", default=True, type=bool, choices=[True, False], help="create hits3 plots"
    )
    parser.add_argument(
        "--plt_hits1", default=True, type=bool, choices=[True, False], help="create hits1 plots"
    )
    parser.add_argument(
        "--dataset", default="ICEWS14", choices=["ICEWS05-15", "ICEWS14"],
        help="Temporal Knowledge Graph Dataset"
    )
    parser.add_argument(
        "--model", default="VectorTransE", choices=["NaiveTransE", "VectorTransE"], help="Temporal Knowledge Graph Embedding Model"
    )
    parser.add_argument(
        "--regularizer", default="N3", choices=["N3", "F2"], help="Regularizer"
    )
    parser.add_argument(
        "--reg", default=0.1, type=float, help="Regularization weight"
    )
    parser.add_argument(
        "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adam", help="Optimizer"
    )
    parser.add_argument(
        "--max_epochs", default=1000, type=int, help="Maximum number of epochs to train for"
    )
    parser.add_argument(
        "--patience", default=3, type=int, help="Number of epochs before early stopping"
    )
    parser.add_argument(
        "--dyn_lr", default=False, type=bool, help="Adjust the learning rate"
    )
    parser.add_argument(
        "--valid", default=1, type=int, help="Number of epochs before validation"
    )
    parser.add_argument(
        "--valid_batch", default=10, type=int, help= "Number of validation batches"
    )
    parser.add_argument(
        "--rank", default=100, type=int, help="Embedding dimension"
    )
    parser.add_argument(
        "--batch_size", default=1000, type=int, help="Batch size"
    )
    parser.add_argument(
        "--neg_sample_size", default=10, type=int, help="Negative sample size, -1 to not use negative sampling"
    )
    parser.add_argument(
        "--init_size", default=0.125, type=float, help="Initial embeddings' scale"
    )
    parser.add_argument(
        "--learning_rate", default=.001, type=float, help="Learning rate"
    )
    parser.add_argument(
        "--bias", default="constant", type=str, choices=["constant", "learn", "none"],
        help="Bias type (none for no bias)"
    )
    parser.add_argument(
        "--debug", default=True, type= bool, help="Faster processing pipeline"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Device for calculations"
    )
    parser.add_argument(
        "--margin", default=1, type=float, help="Margin for distance-based losses"
    )
    parser.add_argument(
        "--num_save_files", default=999999, type=int, help="Decide, how many trainings should be saved"
    )

    return parser.parse_args()
