# -*- coding: utf-8 -*-
from lib.pipeline.handler.previewer import Previewer
from lib.pipeline.handler.trainer import Trainer
from lib.pipeline.handler.exporter import Exporter


class Pipeline:
    def __init__(self, conf_factory):
        self._conf_factory = conf_factory
        self._logger = conf_factory.logger

    def run(self):
        self._logger.info('start to run pipeline')
        self._pipeline().handle(self._conf_factory)

    @staticmethod
    def _pipeline():
        previewer = Previewer()
        trainer = Trainer()
        exporter = Exporter()
        previewer.set_next(trainer).set_next(exporter)
        return previewer
