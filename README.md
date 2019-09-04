# Experiment Code for MultiPath

## Instructions

1. Download the data by running `python3 download.py` in the `data/` directory (Data courtesy of [MINERVA](https://github.com/shehzaadzd/MINERVA)),
1. Run the experiments by picking any of the `run_*.sh` scripts with `./run_*.sh` or `bash run_*.sh`.

## Notes

- You can get a nicely formatted explanation of all of the options by running `python3 main.py --help` in the root directory, or read through the `get_config()` method in `main.py`,
- To see tensorboards, go to `outputs/tensorboards` and run `tensorboard --logdir=FILE_NAME.tb --host=CUSTOM_HOST --port=CUSTOM_PORT`. Port forwarding can be done with `ssh -L BROWSER_PORT:HOST_NAME:SERVER_PORT -N USER_NAME@SERVER`,
