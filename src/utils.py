"""
This file contains some helper functions and stuff

Course: Deep Learning
Date: 28.03.2022
Group: 150
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import os
import numpy as np


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
    Basic timer class to measure different processes easily
    """

    # An internal storage for different processes to measure
    processes = None

    def __init__(self):
        """
        Initializes timer.
        """
        # Saves all processes currently in use.
        self.processes = {}

    def start(self, process):
        """
        Starts a process specific timer
        """
        assert isinstance(process, str)
        self.processes[process] = time.time()

    def get_time(self, process):
        """
        Get time of global timer or process specific timer.

        :param process: the process of which to obtain the timer from
        """
        start_time = self.processes.pop(process, None)
        if start_time is None:
            raise KeyError('Process not found.')
        else:
            return np.round(time.time() - start_time, 2)

    def reset(self, process):
        """
        Reset timer of a process.

        :param process: the process of which to reset the timer from
        """
        self.processes[process] = None

    def reset_all(self):
        """
        Resets all timers
        """
        self.processes = {}


class Logger:
    """
    Class that logs messages.

    TODO: integrate status bar in logger and overwrite last row in out when current status bar
    """

    # Stores the path of the log file
    path = ""

    # Attribute stores whether there is a progressbar currently
    status_bar_active = False
    last_status_bar = None

    # Whether or whether not the logger is initialized
    initialized = False

    @staticmethod
    def initialize(log_name=None, **kwargs):
        """Initialize method

        Parameters
        ----------
        path: str
            path to directory in which log file will be created
        file_name: str
            name of the log file that will be created

        """

        # Obtain the base path at looking at the parent of the parents parent
        base_path = Path(__file__).parent.parent.resolve()

        # Create log path
        log_path = os.path.join(base_path, 'logs')

        # Check if log dir exists, if not create
        Path(log_path).mkdir(parents=True, exist_ok=True)

        # Save variables
        if log_name is not None and log_path is not None:
            # Define path
            Logger.path = os.path.join(log_path, log_name + '.txt')
            # Create file if it does not exist yet
            if not os.path.isfile(Logger.path):
                open(Logger.path, 'w+')
        else:
            Logger.path = None

        # Set initialized
        Logger.initialized = True

    @staticmethod
    def clear():
        """
        Clears the log file

        :return:
        """

        # Clears the file
        with open(Logger.path, 'w') as file:
            file.write("")

    @staticmethod
    def log(message: str, type: str = "", in_cli: bool = True):
        """
        Writes a log message to log file. Message will also be printed in terminal.

        :param message: message to be logged
        :param type: the type of the message
        :param in_cli: whether to also print the message
        """

        # Check whether logger is initialized
        if not Logger.initialized:
            raise Exception("ERROR: Logger is not initialized")

        if in_cli:
            Logger.out(message, type)

        if Logger.path is not None:
            with open(Logger.path, 'a+') as file:
                if message != '':
                    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    log_message = Logger._get_content(message, type, time_str)
                    file.write(log_message)
                file.write('\n')

    @staticmethod
    def out(message: str, type: str):
        """
        Only writes message to terminal for debugging purposes.

        :param message: message to be logged
        :param type: the type of the message
        """

        # Check whether logger is initialized
        if not Logger.initialized:
            raise Exception("ERROR: Logger is not initialized")

        # Get the CLI mssage to display
        cli_message = Logger._get_content(message, type)

        if 'Epoch took' in cli_message:
            a = 0

        if Logger.status_bar_active:
            sys.stdout.write("\r" + cli_message)
            sys.stdout.flush()
            print("")
            if Logger.last_status_bar is not None:
                sys.stdout.write(Logger.last_status_bar)
                sys.stdout.flush()
        else:
            print(cli_message)

    @staticmethod
    def print_status_bar(title: str = "done", done: float = 0, bar_width: int = 50):
        """
        This function shows a "running" status bar to visualize progress

        :param title: the title that shall be displayed besides the bar
        :param done: the percentage of the bar
        :param bar_width: how wide the bar shall be
        """

        # Check whether logger is initialized
        if not Logger.initialized:
            raise Exception("ERROR: Logger is not initialized")

        Logger.status_bar_active = True
        if done == 100.0:
            Logger.last_status_bar = "\r|" + bcolors.OKBLUE + ('.' * bar_width) + bcolors.ENDC + "| 100% " + title
        else:
            status_string = "|" + bcolors.OKBLUE
            state = "d"
            for j in range(bar_width):
                nextstate = state
                if int(done) >= ((j / bar_width) * 100):
                    status_string += "."
                else:
                    if state == 'd':
                        nextstate = 'nd'
                        status_string += bcolors.FAIL
                    status_string += "."
                state = nextstate
            status_string += bcolors.ENDC
            status_string += "| "
            Logger.last_status_bar = "\r" + status_string + str(round(done, 2)) + "% " + title

        sys.stdout.write(Logger.last_status_bar)
        if done != 100.0:
            sys.stdout.flush()
        else:
            print("")

    @staticmethod
    def end_status_bar():
        """
        This function must be called after ending a status bar progress
        """

        # Check whether logger is initialized
        if not Logger.initialized:
            raise Exception("ERROR: Logger is not initialized")

        Logger.status_bar_active = False
        # print("")

    @staticmethod
    def _get_content(message: str, type: str, log_stamp: str = None):
        """

        :param message: the raw message to log or print
        :param type: the type of the message
        :param log_stamp: when passed, it returns a log string
        :return: a string for cli logging or writing to rtf
        """

        # Check whether the type is known
        if type in ['ERROR', 'WARNING', 'INFO', 'SUCCESS', '']:

            # Create prephase when log stamp is given
            if log_stamp is not None:

                # Create a log message
                if type == 'INFO':
                    log_message = log_stamp + "     INFO    \t" + message
                elif type == 'WARNING':
                    log_message = log_stamp + "     WARN    \t" + message
                elif type == 'ERROR':
                    log_message = "\n" + log_stamp + "     ERROR   \t" + message + "\n"
                elif type == 'SUCCESS':
                    log_message = log_stamp + "     SUCCESS \t" + message
                else:
                    log_message = log_stamp + "     INFO    \t" + message

                return log_message

            else:

                # Print loading message
                if type == 'INFO':
                    cli_message = bcolors.OKBLUE + "INFO    " + message + bcolors.ENDC
                elif type == 'WARNING':
                    cli_message = bcolors.WARNING + "WARN    " + message + bcolors.ENDC
                elif type == 'ERROR':
                    cli_message = bcolors.FAIL + "ERROR   " + message + bcolors.ENDC
                elif type == 'SUCCESS':
                    cli_message = bcolors.OKGREEN + "SUCCESS " + message + bcolors.ENDC
                else:
                    cli_message = "LOG     " + message

                return cli_message

        else:

            # Notify that the output type is unknown
            raise ValueError("ERROR: Logger out function does not recognize the message type")