"""
This file contains some helper functions and stuff

Course: Deep Learning
Date: 28.03.2022
Group: 150
"""

import sys
import time
from datetime import datetime
import os


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


class Timer:
    """
    Basic timer class.
    """

    def __init__(self):
        """
        Initializes timer.
        """
        # The start time of the global timer.
        self.start_time = None
        # Saves all processes currently in use.
        self.processes = {}

    def start(self, process=None):
        """
        Starts a global timer or a process specific timer.

        Parameters
        ----------
        process: str
            process that the timer should time, if None global
            time will be used (optional)

        """
        if process is not None:
            assert isinstance(process, str)
            self.processes[process] = time.time()
        else:
            self.start_time = time.time()

    def get_time(self, process=None):
        """
        Get time of global timer or process specific timer.

        Parameters
        ----------
        process: str
            process that you want the current time of, if None
            global time will be returned (optional)

        """
        if process is not None:
            start_time = self.processes.pop(process, None)
            if start_time is None:
                raise KeyError('Process not found.')
            else:
                return np.round(time.time() - start_time, 2)
        else:
            if self.start_time is None:
                raise Exception('Timer has not been started yet.')
            return np.round(time.time() - self.start_time, 2)

    def reset(self, process=None):
        """
        Reset global timer or timer of a process.

        Parameters
        ----------
        process: str
            process of which timer is to be reset, if None
            global time is reset (optional)

        """
        if process is not None:
            self.processes[process] = None
        else:
            self.start_time = None

    def reset_all(self):
        """
        Resets global timer and all processes.
        """
        self.start_time = None
        self.processes = {}


class Logger:
    """
    Class that logs messages.
    """

    def __init__(self, path=None, file_name=None, **kwargs):
        """Initialize method

        Parameters
        ----------
        path: str
            path to directory in which log file will be created
        file_name: str
            name of the log file that will be created

        """
        # Save variables
        if file_name is not None and path is not None:
            # Define path
            self.path = os.path.join(path, file_name + '.txt')
            # Create file if it does not exist yet
            if not os.path.isfile(self.path):
                open(self.path, 'w+')
        else:
            self.path = None

    def write(self, message: str):
        """
        Writes a log message to log file. Message will also be printed in terminal.

        Parameters
        ----------
        message: str
            message to be logged

        """
        message = str(message)
        if self.path is not None:
            with open(self.path, 'a+') as file:
                if message != '':
                    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    file.write(time_str + '         ' + message)
                file.write('\n')
        print(message)

    def debug_write(self, message: str):
        """
        Only writes message to terminal for debugging purposes.

        Parameters
        ----------
        message: str
            message to be logged

        """
        print(message)


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