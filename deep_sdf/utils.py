#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import torch


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def decode_sdf(decoder, code_input, queries, on_data):
    if code_input is None:
        sdf = decoder(queries.squeeze(0))
        sdf = sdf.view(-1, 1)
    else:
        try:
            with torch.no_grad():
                sdf = decoder(queries.cuda(), on_data.cuda(), code_input.cuda(), predict_sp=True)
                N = queries.shape[1]
                sdf = sdf[:, :N, :].view(-1, 1)
        except:
            raise RuntimeError("Failed to decode SDF")

    return sdf
