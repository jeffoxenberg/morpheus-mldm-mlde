import argparse
import sys

from determined.common.api.errors import NotFoundException
from determined.experimental import client
from determined.common.experimental.checkpoint import Checkpoint
from determined.common.experimental.model import ModelVersion
from git import Repo


def get_or_create_model(model: str) -> client.Model:
    """
    Get the model if it exists, if not, and the --create flag is set, create it, and then return it.
    :param model: a Model object representing the model to get or create
    :return: a Model object
    """
    try:
        m = client.get_model(model)
    except NotFoundException as e:
        if args.create:
            print('model not found, creating')
            m = client.create_model(model)
        else:
            sys.exit('model not found and create flag not set, exiting')
    return m


def run_experiment(config: str) -> Checkpoint:
    """
    Run experiment and return the top checkpoint if the experiment is successful.  If it fails, exit.
    :param config: MLDE experiment config yaml
    :return: Checkpoint object representing the top checkpoint for the successfully-run experiment
    """
    exp = client.create_experiment(config=config, model_dir='model')  # exp.id
    status = exp.wait()
    if status.COMPLETED.value == 'STATE_COMPLETED':
        best_checkpoint = exp.top_checkpoint()  # best_checkpoint.uuid
    else:
        sys.exit(f'Error: {status.COMPLETED.value}')
    return best_checkpoint


def register_or_return_version(model: client.Model, checkpoint: Checkpoint) -> ModelVersion:
    """
    Gets the model's latest version, and only stores the best checkpoint from the latest experiment if it's metric is
    better than the existing model version.
    :param model: The model to check and register
    :param checkpoint: The new checkpoint to check against the model
    :return: ModelVersion representing the best model between latest experiment and existing model
    """
    mv = model.get_version()  # gets latest version of model

    existing_best = mv.checkpoint.training.validation_metrics['avgMetrics']['val_f1_accuracy']
    new_best = checkpoint.training.validation_metrics['avgMetrics']['val_f1_accuracy']

    if new_best > existing_best:
        mv = model.register_version(checkpoint.uuid)
        model.add_metadata({'deployed': False})  # model.metadata['deployed'] to read
    return mv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master',
                        required=True,
                        help='MLDE Master')
    parser.add_argument('--model',
                        required=True,
                        help='Model name')
    parser.add_argument('--create',
                        required=False,
                        action='store_true',
                        help="Create model if it doesn't already exist")
    parser.add_argument('--repo',
                        required=True,
                        help='Git repo to pull code from')
    parser.add_argument('--config',
                        required=True,
                        help='Config to launch')
    args = parser.parse_args()

    repo = Repo.clone_from(args.repo, 'model')

    client.login(master=args.master, user='determined')

    m = get_or_create_model(args.model)

    ckpt = run_experiment(args.config)

    mv = register_or_return_version(m, ckpt)

    print(mv)
