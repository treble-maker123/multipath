import logging
import sys
from argparse import Namespace, ArgumentParser
from typing import Tuple, List

from tensorboardX import SummaryWriter

from main_helpers import add_experiment_config, add_model_config, add_rgcn_config


def _get_config() -> Tuple[Namespace, List[str]]:
    parser = ArgumentParser()
    parser = add_experiment_config(parser)
    parser = add_model_config(parser)
    parser = add_rgcn_config(parser)

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
