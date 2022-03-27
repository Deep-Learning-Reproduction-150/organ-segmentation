"""
This file contains some helper functions and stuff

Course: Deep Learning
Date: 28.03.2022
Group: 150
"""

import sys


class bcolors:
    """
    This class contains colors for the terminal
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_status_bar(title: str = "done", done: float = 0):
    if done == 100.0:
        sys.stdout.write(
            "\r|" + bcolors.OKCYAN + "...................................................................................................." + bcolors.ENDC + "| 100% written")
        print("")
    else:
        status_string = "|" + bcolors.OKCYAN
        state = "d"
        for j in range(100):
            nextstate = state
            if int(done) >= j:
                status_string += "."
            else:
                if state == 'd':
                    nextstate = 'nd'
                    status_string += bcolors.FAIL
                status_string += "."
            state = nextstate
        status_string += bcolors.ENDC
        status_string += "| "
        sys.stdout.write("\r" + status_string + str(round(done, 2)) + "% " + title)
        sys.stdout.flush()