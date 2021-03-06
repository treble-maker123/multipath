import sys
from time import time
from typing import Dict, Type

import torch
import yaml

from config import config, unknown, logger, tensorboard
from lib.engines import Engine, RGCNEngine, PathTransformLinkPredictEngine

ENGINE_TYPES: Dict[str, Type[Engine]] = {
    "rgcn": RGCNEngine,
    "path-transform-link-predict": PathTransformLinkPredictEngine
}

# https://github.com/pytorch/pytorch/issues/1485
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    # Setup PYTHONPATH
    sys.path.append("...")

    engine_type = ENGINE_TYPES[config.engine]
    engine = engine_type()

    if config.interactive:
        logger.info("Setup completed, entering interactive mode...")
    else:
        logger.info(f"----- START (Job ID: {config.run_id}) -----\n")
        logger.info(f"Following configurations are used for this run:\n"
                    f"{yaml.dump(vars(config), default_flow_style=False)}"
                    f"Unknown arguments received: {unknown}.")

        start_time = time()
        engine.run()
        end_time = time()

        tensorboard.close()

        logger.info(f"Experiment finished in {round(end_time - start_time)} seconds.")
        logger.info(f"----- END ({config.run_id}) -----")
