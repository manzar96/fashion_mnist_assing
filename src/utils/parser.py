import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Main arguments")

    parser.add_argument(
        "--bs", "--batch-size", type=int, default=32, help="Training/eval "
                                                           "batch size"
    )
    parser.add_argument(
        "--es", "--epochs", type=int, default=20, help="epochs to train model"
    )
    parser.add_argument(
        "--patience",  type=int, default=5, help="patience for early stopping"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate used"
    )
    parser.add_argument(
        "--clip", type=float, default=None, help="clipping for grads"
    )
    parser.add_argument(
        "--modelckpt",
        type=str,
        help="Checkpoint path to save model",
    )
    parser.add_argument(
        "--loadckpt",
        type=str,
        help="Checkpoint to load pretrained model",
    )
    parser.add_argument(
        "--kfold_dir",
        type=str,
        help="Checkpoint to load pretrained model for kfold",
    )
    parser.add_argument(
        '--skip_validation', dest='skip_validation',
        help='Whether to skip validation',
         default=False,action='store_true'
    )
    parser.add_argument(
        '--not_use_early_stopping', dest='not_use_early_stopping',
        help='Whether to dont use early_stopping',
         default=False,action='store_true'
    )
    return parser.parse_args()