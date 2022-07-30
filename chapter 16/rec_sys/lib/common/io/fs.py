# -*- coding: utf-8 -*-


class FS:
    @staticmethod
    def exists(path):
        raise NotImplementedError('FS not implement exists')

    @staticmethod
    def ls(path):
        raise NotImplementedError('FS not implement ls')

    @staticmethod
    def get(src, dst, recursive=False, **kwargs):
        raise NotImplementedError('FS not implement get')

    @staticmethod
    def put(src, dst):
        raise NotImplementedError('FS not implement put')
