# -*- coding: utf-8 -*-
import os
import json
from lib.data import reader
from lib import flags as _flags
from lib import model_fn
from tensorflow.compat.v1 import app
from tensorflow.compat.v1 import logging
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import estimator
from tensorflow.compat.v1.distribute import experimental


def _tf_config(_flags):
    tf_config = dict()
    ps = ['localhost:2220']
    chief = ['localhost:2221']
    worker = ['localhost:2222']
    evaluator = ['localhost:2223']

    cluster = {
        'ps': ps,
        'chief': chief,
        'worker': worker,
        'evaluator': evaluator
    }

    task = {
        'type': _flags.type,
        'index': _flags.index
    }

    tf_config['cluster'] = cluster
    tf_config['task'] = task

    if _flags.type == 'chief':
        _flags.__dict__['worker_index'] = 0
    elif _flags.type == 'worker':
        _flags.__dict__['worker_index'] = 1

    _flags.__dict__['num_workers'] = len(worker) + len(chief)

    _flags.__dict__['device_filters'] = ["/job:ps", f"/job:{_flags.type}/task:{_flags.index}"]

    return tf_config


def _run_config(flags):
    cpu = os.cpu_count()
    session_config = ConfigProto(
        device_count={'GPU': flags.gpu or 0,
                      'CPU': flags.cpu or cpu},
        inter_op_parallelism_threads=flags.inter_op_parallelism_threads or cpu // 2,
        intra_op_parallelism_threads=flags.intra_op_parallelism_threads or cpu // 2,
        allow_soft_placement=True)

    strategy = experimental.ParameterServerStrategy()  # 3

    return {
        'save_summary_steps': int(flags.save_summary_steps),
        'save_checkpoints_steps': int(flags.save_checkpoints_steps),
        'keep_checkpoint_max': int(flags.keep_checkpoint_max),
        'log_step_count_steps': int(flags.log_step_count_steps),
        'session_config': session_config,
        'train_distribute': strategy,
        'eval_distribute': strategy
    }


def _build_run_config(flags):
    sess_config = _run_config(flags)
    return estimator.RunConfig(**sess_config)


def main(argv):
    flags = argv[0]
    tf_config = _tf_config(flags)  # 1
    # 分布式需要 TF_CONFIG 环境变量
    os.environ['TF_CONFIG'] = json.dumps(tf_config)  # 2
    run_config = _build_run_config(flags)

    _params = {}
    _params.update(flags.__dict__)

    model = estimator.Estimator(
        model_fn=model_fn,
        model_dir=str(flags.checkpoint_dir),
        config=run_config,
        params=_params
    )

    train_spec = estimator.TrainSpec(input_fn=lambda: reader.input_fn(mode='train', flags=flags),
                                     max_steps=1000)  # 4

    eval_spec = estimator.EvalSpec(
        input_fn=lambda: reader.input_fn(mode='eval', flags=flags),
        steps=int(flags.eval_steps),
        throttle_secs=int(flags.eval_throttle_secs)
    )
    estimator.train_and_evaluate(model, train_spec, eval_spec)

    if __name__ == '__main__':
        logging.set_verbosity(logging.FATAL)
    app.run(main=main, argv=[_flags])
