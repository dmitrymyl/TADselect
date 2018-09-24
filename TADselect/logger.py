'''
Logging setup
'''

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
TADselect_logger = logging.getLogger('TADselect')
TADselect_logger.setLevel(logging.getLevelName('DEBUG'))


def set_verbosity(level):
    TADselect_logger.setLevel(logging.getLevelName(level))

__all__ = ('TADselect_logger', 'set_verbosity')
