# -*- coding: utf-8 -*-
import os
import argparse
import platform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"


class Flags:
    def __init__(self):
        self._flags = self._parse()

    @property
    def flags(self):
        return self._flags

    def _parse(self):
        if platform.system().lower() == 'windows':
            from pathlib import WindowsPath as SystemPath
        else:
            from pathlib import PosixPath as SystemPath

        _project_dir = SystemPath(__file__).absolute().parents[1]

        _parser = argparse.ArgumentParser()
        self._add_arguments(_parser)
        _flags, _unknowns = _parser.parse_known_args()
        if _unknowns:
            for _unknown in _unknowns:
                _k, _v = _unknown.split('=')
                _k = _k.strip().lstrip('--')
                _v = _v.strip()
                _flags.__dict__[_k] = _v
        _flags.__dict__['project_dir'] = _project_dir
        _flags.__dict__['model_dir'] = _project_dir.joinpath(_flags.model_dir)
        _flags.__dict__['export_dir'] = _project_dir.joinpath(_flags.export_dir)

        return _flags

    @staticmethod
    def _add_arguments(parser):
        parser.register("type", "bool", lambda v: v.lower() == "true")
        parser.add_argument("--dataset", type=str, help="dataset name")
        parser.add_argument("--preview", type=lambda x: (str(x).lower() == 'true'), default=False,
                            help="preview data.")
        parser.add_argument("--train", type=lambda x: (str(x).lower() == 'true'), default=True,
                            help="train model.")
        parser.add_argument("--export", type=lambda x: (str(x).lower() == 'true'), default=True,
                            help="export model.")
        parser.add_argument("--train_test_split", type=float, default=0.7, help="split train and test set.")
        parser.add_argument("--model_dir", type=str, default="checkpoints", help="model path.")
        parser.add_argument("--model_name", type=str, required=True, help="model name, required.")
        parser.add_argument("--export_dir", type=str, default="exporters", help="")
        parser.add_argument("--learning_rate", type=float, default=0.05, help="learning_rate.")
        parser.add_argument("--decay_rate", type=float, default=0.8, help="learning decay rate.")
        parser.add_argument("--decay_steps", type=int, default=400000, help="learning decay steps.")
        parser.add_argument("--batch_size", type=int, help="batch size.")
        parser.add_argument("--num_epochs", type=int, default=1, help="epoch number.")
        parser.add_argument("--buffer_size", type=int, default=10000, help="buffer size.")
        parser.add_argument("--save_checkpoints_steps", type=int, default=50000, help="save_checkpoints_steps.")
        parser.add_argument("--save_summary_steps", type=int, default=5000, help="save_summary_steps.")
        parser.add_argument("--log_step_count_steps", type=int, default=2000, help="log_step_count_steps.")
        parser.add_argument("--eval_steps", type=int, default=1000, help="eval_steps.")
        parser.add_argument("--max_steps", type=int, default=None, help="max_steps.")
        parser.add_argument("--inter_op_parallelism_threads", type=int,
                            help="number of inter_op_parallelism_threads")
        parser.add_argument("--intra_op_parallelism_threads", type=int,
                            help="number of intra_op_parallelism_threads")
        parser.add_argument("--cpu", type=int, help="number of cpu")
        parser.add_argument("--gpu", type=int, help="number of gpu")
        parser.add_argument("--num_parallel_calls", type=int, help="num parallel calls.")
