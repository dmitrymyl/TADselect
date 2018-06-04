import os
import subprocess
import numpy as np
import cooler
import lavaburst
import pandas as pd
import time

from .logger import logger

######## Basic utils to run linux commands #########

def call_and_check_errors(command):

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=True, executable='/bin/bash')
    (stdout, stderr) = proc.communicate()
    logger.info("Check stdout: {}".format(stdout))
    if stderr:
        logger.info("Stderr is not empty. Might be an error in call_and_check_errors for the command: {}".format(command))
        logger.info("Check stderr: {}".format(stderr))
        return stderr   # Error, very bad!
    else:
        return 0        # No error, great!

def run_command(command, force=False):

    logger.info(command)

    possible_outfile = command.split('>')

    if len(possible_outfile)>1:
        possible_outfile = possible_outfile[-1]
        if os.path.isfile(possible_outfile):
            if force:
                logger.info("Outfile {} exists. It will be overwritten!".format(possible_outfile))
            else:
                raise Exception("Outfile {} exists. Please, delete it, or use force=True to overwrite it.".format(possible_outfile))

    cmd_bgn_time = time.time()
    is_err = call_and_check_errors(command)
    cmd_end_time = time.time()

    return is_err
