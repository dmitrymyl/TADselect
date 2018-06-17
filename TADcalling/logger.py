'''
Logging setup
'''

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
TADcalling_logger = logging.getLogger('TADcalling')
TADcalling_logger.setLevel(logging.getLevelName('DEBUG'))


def set_verbosity(level):
    TADcalling_logger.setLevel(logging.getLevelName(level))

__all__ = ('TADcalling_logger', 'set_verbosity')
