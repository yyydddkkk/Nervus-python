"""The one built-in capability whose OS resource is owned by the supervisor."""


import os


class ProcessRunner:
    def __init__(self, *, output_limit: int = 65536):
        if type(output_limit) is not int or output_limit < 0:
            raise ValueError('output_limit must be a nonnegative byte count per stream')
        self.output_limit = output_limit

    async def initialize(self):
        if os.name != 'posix' or not all(hasattr(os, name) for name in ('waitid', 'WNOWAIT', 'killpg')):
            raise RuntimeError('ProcessRunner requires POSIX process groups and waitid/WNOWAIT')

    async def __call__(self, argv):
        raise RuntimeError('ProcessRunner must be invoked through a Nervus capability')

    async def close(self):
        pass


def validate_argv(argv):
    if (type(argv) is not list or not argv
            or any(type(arg) is not str or '\0' in arg for arg in argv)
            or not argv[0]):
        raise ValueError('argv must be a nonempty list of strings without NUL bytes')
    return list(argv)
