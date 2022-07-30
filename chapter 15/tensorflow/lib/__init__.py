# -*- coding: utf-8 -*-

import os
import platform
import argparse
from lib.common import args
from model.estimator import Estimator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"

if platform.system().lower() == 'windows':
    from pathlib import WindowsPath as SystemPath
else:
    from pathlib import PosixPath as SystemPath

PROJECT_DIR = SystemPath(__file__).absolute().parents[1]

parser = argparse.ArgumentParser()
args.add_arguments(parser)
flags, un_parsed = parser.parse_known_args()
if un_parsed:
    for un in un_parsed:
        _k, _v = un.split('=')
        _k = _k.strip().lstrip('--')
        _v = _v.strip()
        flags.__dict__[_k] = _v


def model_fn(features, labels, mode, params):
    return Estimator(features, labels, mode, params).model_fn()
