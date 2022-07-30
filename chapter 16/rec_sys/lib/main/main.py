import tensorflow as tf
from lib import Flags
from lib.conf.conf_factory import ConfFactory
from lib.pipeline.pipeline import Pipeline


def main(argv):
    flags, = argv
    conf_factory = ConfFactory(flags)
    pipeline = Pipeline(conf_factory)
    pipeline.run()


if __name__ == '__main__':
    tf.app.run(main=main, argv=[Flags().flags])
