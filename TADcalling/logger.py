'''
Logging setup
'''

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('TADcalling')
logger.setLevel(logging.getLevelName('DEBUG'))


def set_verbosity(level):
    logger.setLevel(logging.getLevelName(level))

__all__ = ('logger',  'set_verbosity')
