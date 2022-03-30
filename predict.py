"""
    main module for prediction util
"""

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
#from src.logs import setup_logging
#from src.logs import LOGGER_YAML_DEFAULT
from src.model import ModelPredict
from src.transforms import val_transforms
from src.constants import NAMES
import torch




def setup_parser(parser):
    """Setups parser"""
    # parser.set_defaults(callback=callback_analytics)

    parser.add_argument(
        "--data_dir",
        "-dd",
        required=True,
        default=None,
        help="path to data",
        metavar="FPATH",
    )

    # parser.add_argument(
    #     "--config",
    #     "-c",
    #     required=True,
    #     default=None,
    #     help="path to config model file",
    #     metavar="FPATH",
    # )


    parser.add_argument(
        "--model_dir",
        "-md",
        required=True,
        default=None,
        help="path to model",
        metavar="FPATH",
    )

    parser.add_argument(
        "--image_name",
        "-in",
        required=True,
        default=None,
        help="image file name",
        metavar="FPATH",
    )

    # parser.add_argument(
    #     "--transformer",
    #     "-tp",
    #     required=True,
    #     default=None,
    #     help="path to transformer",
    #     metavar="FPATH",
    # )
    #
    # parser.add_argument(
    #     "--output",
    #     "-sp",
    #     required=True,
    #     default=None,
    #     help="path to saved predictions",
    #     metavar="FPATH",
    # )


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="load data and model and make predict",
        description="instrument for making prediction",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    setup_parser(parser)
    arguments = parser.parse_args()
    # if not os.path.exists(arguments.output):
    #     os.mkdir(arguments.output)

    #setup_logging(LOGGER_YAML_DEFAULT, os.path.join(arguments.output, "predict.log"))
    # arguments.callback(arguments)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model_to_predict = ModelPredict(arguments.model_dir, arguments.data_dir, device, val_transforms)
    # image_file = glob.glob(os.path.join(data_dir, image_name))
    # image_name = np.random.choice(image_name)
    model_to_predict.load_model("best_resnet.pth")
    prediction = model_to_predict.predict(arguments.image_name)

    
