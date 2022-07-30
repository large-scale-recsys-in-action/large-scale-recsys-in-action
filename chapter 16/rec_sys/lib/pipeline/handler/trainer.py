# -*- coding: utf-8 -*-
import os
from tensorflow import estimator
from tensorflow import ConfigProto
from lib.pipeline.handler.handler import Handler
from lib.dataset import factory


class Trainer(Handler):
    def __init__(self):
        super(Trainer, self).__init__()

    def _handle(self, conf_factory):
        if conf_factory.flags.train:
            conf_factory.logger.info("stage: training")
            self._conf_factory = conf_factory
            self._flags = conf_factory.flags
            self._model_conf = conf_factory.model_conf
            self._feature_conf = conf_factory.feature_conf
            self._logger = conf_factory.logger

            model_name = self._flags.model_name
            model_dir = str(self._flags.model_dir)

            model_fn = self._build_model_fn(model_name)
            _params = self._build_params(self._conf_factory)
            run_config = self._build_run_config()
            model = estimator.Estimator(model_fn=model_fn,
                                        model_dir=model_dir,
                                        config=run_config,
                                        params=_params)
            self.train_and_eval(model)

    def train_and_eval(self, model):
        train_spec = estimator.TrainSpec(
            input_fn=lambda: factory.input_fn(self._conf_factory, mode='train'),
            max_steps=self._model_conf['max_steps'])
        eval_spec = estimator.EvalSpec(
            input_fn=lambda: factory.input_fn(self._conf_factory, mode='eval'),
            steps=self._model_conf['eval_steps'],
            throttle_secs=self._model_conf['eval_throttle_secs'])
        estimator.train_and_evaluate(model, train_spec, eval_spec)
        model.evaluate(input_fn=lambda: factory.input_fn(self._conf_factory, mode='eval'),
                       steps=5 * self._model_conf['eval_steps'])

    def _build_session_config(self):
        device_filters = []
        cpus = os.cpu_count()
        session_config = ConfigProto(
            device_count={'CPU': self._flags.cpu or cpus},
            inter_op_parallelism_threads=cpus // 2,
            intra_op_parallelism_threads=cpus // 2,
            device_filters=device_filters,
            allow_soft_placement=True)

        return {
            'save_summary_steps': self._model_conf['save_summary_steps'],
            'save_checkpoints_steps': self._model_conf['save_checkpoints_steps'],
            'keep_checkpoint_max': self._model_conf['keep_checkpoint_max'],
            'log_step_count_steps': self._model_conf['log_step_count_steps'],
            'session_config': session_config
        }

    def _build_run_config(self):
        sess_config = self._build_session_config()
        return estimator.RunConfig(**sess_config)
