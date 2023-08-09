import multiprocessing
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
import multiprocessing.pool
import time

from random import randint


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    # Process = NoDaemonProcess

    def Process(self, *args, **kwds):
        proc = super(MyPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess

        return proc
#################old ###################################################3
# import multiprocessing
# # We must import this explicitly, it is not imported by the top-level
# # multiprocessing module.
# # https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
# import multiprocessing.pool
# import time
#
# from random import randint

# class NoDaemonProcess(multiprocessing.Process):
#     # make 'daemon' attribute always return False
#     def _get_daemon(self):
#         return False
#     def _set_daemon(self, value):
#         pass
#     daemon = property(_get_daemon, _set_daemon)
#
# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.
# class MyPool(multiprocessing.pool.Pool):
#     Process = NoDaemonProcess