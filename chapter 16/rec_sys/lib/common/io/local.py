# -*- coding: utf-8 -*-
import os
from lib.common.io.fs import FS
from shutil import copy2


class Local(FS):
    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @staticmethod
    def ls(path):
        return [os.path.abspath(os.path.join(path, f)) for f in os.listdir(path)]

    @staticmethod
    def get(src, dst, recursive=False, **kwargs):
        return copy2(src, dst)

    @staticmethod
    def put(src, dst):
        return copy2(src, dst)
