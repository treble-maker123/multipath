import uuid
from argparse import ArgumentParser


def add_rgcn_config(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--num-rgcn-layers", type=int,
                        default=2, help="Number of hidden layers for encoders.")
    parser.add_argument("--num-bases", type=int,
                        default=100, help="Number of bases or blocks for RGCN.")
    parser.add_argument("--graph-sample-size", type=int,
                        default=30000, help="Number of edges to sample to build the training graph each epoch.")
    parser.add_argument("--train-graph-split", type=float,
                        default=0.5, help="How much of the training graph is split to be used as training data.")
    parser.add_argument("--loop-dropout", type=float,
                        default=0.2, help="Dropout rate used on self loop.")
    parser.add_argument("--negative-sample-factor", type=int,
                        default=10, help="How much negative sampling to apply to the training data.")
    parser.add_argument("--grad-norm", type=float,
                        default=1.0, help="norm to clip gradient to.")

    return parser


def add_model_config(parser: ArgumentParser):
    parser.add_argument("--hidden-dim", type=int,
                        default=500, help="Number of units for the node/entity representation.")
    parser.add_argument("--learn-rate", type=float,
                        default=0.01, help="Learning rate for the model.")
    parser.add_argument("--weight-decay", type=float,
                        default=0.0, help="Weight decay or L2 norm on the model.")
    parser.add_argument("--embedding-decay", type=float,
                        default=0.01, help="Regularization on the node and edge embeddings.")

    return parser


def add_experiment_config(parser):
    parser.add_argument("--interactive", dest="interactive",
                        action="store_true",
                        default=False,
                        help="Whether to run this in interactive mode, where the engine will be setup, but will not "
                             "execute the training loop.")
    parser.add_argument("--run-id", type=str,
                        default=f"DEFAULT_{uuid.uuid4().hex.upper()[0:4]}",
                        help="A string that uniquely identifies this run.")
    parser.add_argument("--log-level", type=int, default=10,
                        help="Logging level, see Python logging module for deatils.")
    parser.add_argument("--log-to-file", dest="log_to_file",
                        action="store_true",
                        default=False,
                        help="Whether to write logs to a file.")
    parser.add_argument("--log-to-stdout", dest="log_to_stdout",
                        action="store_true",
                        default=False,
                        help="Whether to write logs to a stdout.")
    parser.add_argument("--write-tensorboard",
                        dest="write_tensorboard",
                        action="store_true",
                        default=False,
                        help="Whether to create tensorboard.")
    parser.add_argument("--use-gpu", dest="use_gpu",
                        action="store_true",
                        default=False,
                        help="Whether to use GPU for training.")
    parser.add_argument("--engine", type=str,
                        default="multipath", help="Which engine to use, see ENGINE_TYPES in main.py for a list of "
                                                  "available engines.")
    parser.add_argument("--dataset-path", type=str,
                        default="data/nell-995", help="Path to the dataset.")
    parser.add_argument("--save-model", dest="save_model",
                        action="store_true", default=False,
                        help="Whether to save the best model.")
    parser.add_argument("--saved-model-path", type=str,
                        default="outputs/models", help="Path to the directory where models will be stored.")
    parser.add_argument("--save-result", dest="save_result",
                        action="store_true", default=False,
                        help="Whether to save the test result.")
    parser.add_argument("--saved-result-path", type=str,
                        default="outputs/results", help="Path to the directory where results will be stored.")
    parser.add_argument("--data-size", type=int,
                        default=-1, help="Size of custom data generated from the graph for training and dev, set to -1 "
                                         "to use existing data.")
    parser.add_argument("--num-epochs", type=int,
                        default=10, help="Number of training epochs.")
    parser.add_argument("--validate-interval", type=int,
                        default=10, help="How many epochs of training to run before running validation on the training "
                                         "set")
    parser.add_argument("--train-batch-size", type=int,
                        default=32, help="Batch size of triplets.")
    parser.add_argument("--test-batch-size", type=int,
                        default=64, help="Batch size of triplets.")

    return parser
