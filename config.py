import logging
import sys
import uuid
from argparse import Namespace, ArgumentParser
from typing import Tuple, List

import torch
from tensorboardX import SummaryWriter


def add_custom_config(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--max-traversal-hops", type=int,
                        default=2, help="Maximum number of hops to make when enumerating paths between two nodes.")
    parser.add_argument("--max-paths", type=int,
                        default=10000, help="Maximum number of paths to consider, if more paths are present, max "
                                            "number of paths are sampled.")

    return parser


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
    parser.add_argument("--rgcn-regularizer", type=str,
                        default="bbd", help="bdd (block-diagonal decomposition) or basis (basis decomposition)")
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
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of workers to use for data loader.")
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
    parser.add_argument("--run-train-during-validate", dest="run_train_during_validate", action="store_true",
                        default=False, help="Whether or not to run through the training set during validation.")
    parser.add_argument("--train-batch-size", type=int,
                        default=32, help="Batch size of triplets.")
    parser.add_argument("--test-batch-size", type=int,
                        default=64, help="Batch size of triplets.")

    return parser


def _get_config() -> Tuple[Namespace, List[str]]:
    parser = ArgumentParser()
    parser = add_experiment_config(parser)
    parser = add_model_config(parser)
    parser = add_rgcn_config(parser)
    parser = add_custom_config(parser)

    known_config, unknown_config = parser.parse_known_args()

    return known_config, unknown_config


config, unknown = _get_config()


def _get_logger(config: Namespace) -> logging.Logger:
    log_handlers = []

    if config.log_to_stdout:
        log_handlers.append(logging.StreamHandler(sys.stdout))

    if config.log_to_file:
        log_handlers.append(
            logging.FileHandler(filename=f"outputs/logs/{config.run_id}.txt"))

    logging.basicConfig(level=config.log_level,
                        handlers=log_handlers,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    logger = logging.getLogger()

    return logger


logger = _get_logger(config)


class SummaryWriterWrapper(object):
    """Wrap around TensorBoard so don't need to wrap functions around if statements.
    """

    def __init__(self, write_tensorboard: bool = False):
        self.write_tensorboard = write_tensorboard

        if self.write_tensorboard:
            self.output_path = f"outputs/tensorboards/{config.run_id}.tb"
            self.summary_writer = SummaryWriter(self.output_path)
        else:
            self.output_path = None
            self.summary_writer = None

    def __getattr__(self, attr):
        if self.write_tensorboard:
            original_attr = self.summary_writer.__getattribute__(attr)

            if callable(original_attr):
                def hooked(*args, **kwargs):
                    result = original_attr(*args, **kwargs)
                    if result == self.summary_writer:
                        return self
                    return result

                return hooked
            else:
                return original_attr
        else:
            return lambda *args, **kwargs: None


tensorboard = SummaryWriterWrapper(config.write_tensorboard)


def _get_configured_device():
    num_gpu = torch.cuda.device_count()

    if config.use_gpu and num_gpu > 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using {device} for training (GPUs: {num_gpu}).")

    return device


configured_device = _get_configured_device()
