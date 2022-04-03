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

    TODO: integrate status bar in logger and overwrite last row in out when current status bar
    """

    # Stores the path of the log file
    path = ""

    # Attribute stores whether there is a progressbar currently
    status_bar_active = False

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

        if in_cli:
            Logger.out(message, type)

        if Logger.path is not None:
            with open(Logger.path, 'a+') as file:
                if message != '':
                    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    file.write(time_str + '         ' + message)
                file.write('\n')

    @staticmethod
    def out(message: str, type: str):
        """
        Only writes message to terminal for debugging purposes.

        :param message: message to be logged
        :param type: the type of the message
        """

        # Check whether the type is known
        if type in ['ERROR', 'WARNING', 'INFO', 'SUCCESS', '']:

            # Print loading message
            if type == 'INFO':
                cli_message = bcolors.OKBLUE + "INFO: " + message + bcolors.ENDC
            elif type == 'WARNING':
                cli_message = bcolors.WARNING + "WARNING: " + message + bcolors.ENDC
            elif type == 'ERROR':
                cli_message = bcolors.FAIL + "ERROR: " + message + bcolors.ENDC
            elif type == 'SUCCESS':
                cli_message = bcolors.OKGREEN + "SUCCESS: " + message + bcolors.ENDC
            else:
                cli_message = "INFO: " + message

            if Logger.status_bar_active:
                sys.stdout.write("\r" + cli_message)
                sys.stdout.flush()
                print("")
            else:
                print(cli_message)

        else:

            # Notify that the output type is unknown
            raise ValueError("ERROR: Logger out function does not recognize the message type")

    @staticmethod
    def print_status_bar(title: str = "done", done: float = 0, bar_width: int = 50):
        """
        This function shows a "running" status bar to visualize progress

        :param title: the title that shall be displayed besides the bar
        :param done: the percentage of the bar
        :param bar_width: how wide the bar shall be
        """
        Logger.status_bar_active = True
        if done == 100.0:
            sys.stdout.write(
                "\r|" + bcolors.OKBLUE + ('.' * bar_width) + bcolors.ENDC + "| 100% " + title)
            print("")
        else:
            status_string = "|" + bcolors.OKBLUE
            state = "d"
            for j in range(bar_width):
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

    @staticmethod
    def end_status_bar():
        """
        This function must be called after ending a status bar progress
        """
        Logger.status_bar_active = False