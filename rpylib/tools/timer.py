"""
Timer decorator taken from "Fluent Python" p205

Adding this decorator to a function allows to time it and displays this information in the console.
"""

import functools
import time


def timer(func):
    @functools.wraps(func)
    def timed(*args, **kwargs):
        t0 = time.time()
        print("Started the " + time.strftime('%d-%b-%y at %H:%M:%S'))
        print('---------------------------------')
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours > 0:
            print('[{:0>2}h{:0>2}m{:0>2}s] {}({}) -> {} '.format(int(hours), int(minutes), int(seconds), name, arg_str,
                                                                 result))
        elif minutes > 0:
            print('[{:0>2}m{:0>2}s] {}({}) -> {} '.format(int(minutes), int(seconds), name, arg_str, result))
        else:
            print('[{:02.5f}s] {}({}) -> {} '.format(seconds, name, arg_str, result))
        return result

    return timed
