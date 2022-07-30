# -*- coding: utf-8 -*-
from lib.pipeline.handler.handler import Handler
from lib.dataset import factory


class Previewer(Handler):
    def __init__(self):
        super(Previewer, self).__init__()

    def _handle(self, conf_factory):
        if conf_factory.flags.preview:
            batch_size = conf_factory.flags.batch_size
            conf_factory.logger.info("stage: previewing dataset")
            conf_factory.flags.batch_size = 1
            conf_factory.logger.info(factory.input_fn(conf_factory, _type='preview'))
            conf_factory.flags.batch_size = batch_size
