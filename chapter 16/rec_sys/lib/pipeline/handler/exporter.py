# -*- coding: utf-8 -*-
import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import estimator
from tensorflow import placeholder
from lib.pipeline.handler.handler import Handler
from lib.feature.feature_factory import FeatureFactory
from tensorflow.compat.v1 import make_tensor_proto
from tensorflow_serving.apis import predict_pb2, model_pb2, prediction_log_pb2


class Exporter(Handler):
    def __init__(self):
        super(Exporter, self).__init__()

    def _handle(self, conf_factory):
        if conf_factory.flags.export:
            self._conf_factory = conf_factory
            self._flags = conf_factory.flags
            self._model_conf = conf_factory.model_conf
            self._feature_conf = conf_factory.feature_conf
            self._slot_feature_map = FeatureFactory(self._feature_conf).slot_feature_map
            self._logger = conf_factory.logger
            self._logger.info("stage: exporting model")
            self._export()

    def _export(self):
        try:
            export_dir = self._export_model()
            self._logger.info('export: {}'.format(export_dir))
            warm_up_dir = export_dir.joinpath('assets.extra')
            warm_up_filename = self._warm_up(warm_up_dir)
            self._logger.info('warm_up_filename: {}'.format(warm_up_filename))
            base_dir = export_dir.parent

        except Exception as e:
            raise RuntimeError('model export ERROR: {}'.format(e))

        return str(base_dir)

    def _export_model(self):
        serving_slots = self._model_conf['serving_slots']

        def serving_input_receiver_fn():
            receiver_tensors = {self._slot_feature_map[slot].name: placeholder(
                dtype=self._tf_type(self._slot_feature_map[slot].d_type),
                shape=(None, None),
                name=self._slot_feature_map[slot].name)
                for slot in serving_slots}

            return estimator.export.build_raw_serving_input_receiver_fn(receiver_tensors)

        model_name = self._flags.model_name
        model_dir = str(self._flags.model_dir)

        model_fn = self._build_model_fn(model_name)
        _params = self._build_params(self._conf_factory)
        model = estimator.Estimator(model_fn=model_fn,
                                    model_dir=model_dir,
                                    params=_params)

        exported_model_dir = model.export_saved_model(str(self._flags.export_dir),
                                                      serving_input_receiver_fn())
        exported_model_dir = pathlib.Path(str(exported_model_dir, 'utf-8'))
        return exported_model_dir

    def _warm_up(self, warm_up_dir):
        def asset_extra():
            _name_value_type = []

            for slot in self._model_conf['serving_slots']:
                column = self._slot_feature_map[slot]
                name = column.name
                d_type = column.d_type
                tf_type = self._tf_type(d_type)

                if d_type == 'string':
                    value = [['']]
                elif d_type == 'float32':
                    value = [[0.0]]
                elif d_type == 'int32':
                    value = [[0]]
                elif d_type == 'int64':
                    value = [[np.int64(0)]]
                else:
                    raise ValueError('data type {} not supported.'.format(d_type))

                _name_value_type.append((name, value, tf_type))

            return _name_value_type

        os.makedirs(warm_up_dir, exist_ok=True)
        warm_up_file = str(warm_up_dir.joinpath('tf_serving_warmup_requests'))

        name_value_type = asset_extra()
        with tf.io.TFRecordWriter(warm_up_file) as writer:
            request = predict_pb2.PredictRequest(
                model_spec=model_pb2.ModelSpec(name=self._flags.model_name, signature_name='serving_default'),
                inputs={_name: make_tensor_proto(_value, dtype=_type) for _name, _value, _type in name_value_type})
            log = prediction_log_pb2.PredictionLog(
                predict_log=prediction_log_pb2.PredictLog(request=request))
            writer.write(log.SerializeToString())

        assert os.path.exists(warm_up_file), 'warm up file: {} missing.'.format(warm_up_file)
        assert os.path.getsize(warm_up_file) > 0, 'warm up file: {} empty.'.format(warm_up_file)

        return warm_up_file
