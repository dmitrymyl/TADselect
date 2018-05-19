'''
Logging setup
'''

import logging

logger = logging.getLogger('TADcalling')
logger.setLevel(logging.getLevelName('DEBUG'))

def set_verbosity(level):
    logger.setLevel(logging.getLevelName(level))

__all__ = ('logger',  'set_verbosity')