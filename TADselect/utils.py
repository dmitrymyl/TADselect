import os
import subprocess
import numpy as np
import cooler
import lavaburst
import pandas as pd
import time

from .logger import TADselect_logger


# Basic utils to run linux commands ###


def call_and_check_errors(command):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            shell=True, executable='/bin/bash')
    (stdout, stderr) = proc.communicate()
    TADselect_logger.info("Check stdout: {}".format(stdout))
    if stderr:
        TADselect_logger.info("Stderr is not empty. Might be an error in call_and_check_errors for the command: %s" % command)
        TADselect_logger.info("Check stderr: %s" % stderr)
        return stderr
    else:
        return 0


def run_command(command, force=False):
    TADselect_logger.info(command)

    possible_outfile = command.split('>')

    if len(possible_outfile) > 1:
        possible_outfile = possible_outfile[-1]
        if os.path.isfile(possible_outfile):
            if force:
                TADselect_logger.info("Outfile %s exists. It will be overwritten!" % possible_outfile)
            else:
                TADselect_logger.error("Outfile %s exists. Please, delete it, or use force=True to overwrite it."
                                        % possible_outfile)

    cmd_bgn_time = time.time()
    is_err = call_and_check_errors(command)
    cmd_end_time = time.time()

    TADselect_logger.info("Command completed: %f" % (cmd_end_time - cmd_bgn_time))

    return is_err
