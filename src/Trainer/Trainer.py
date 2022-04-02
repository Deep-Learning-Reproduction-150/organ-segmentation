"""
This file contains the trainer class which performs jobs

Course: Deep Learning
Date: 03.04.2022
Group: 150
"""

import torch
import math
import json
import os
from src.utils import Logger, Timer, bcolors
from pathlib import Path
from torch.utils.data import random_split, DataLoader
from src.OrganNet25D.network import OrganNet25D
from src.Dataloader.CTDataset import CTDataset


class Trainer:
    """
    This trainer instance performs the training of the network by executing jobs

    TODO:
        - this logic could be extended to multi trainers (using threading)
        - the job type of being will dictate _ functions
        - feel free to add ideas what the trainer should be able to do
        - !!! We should definitely include a "resume" option in the trainer !!!
    """

    # Attribute stores an instance of the network
    model = None

    # An instance of a logger to write into log files (if specified in job)
    logger = None

    # An instance of a timer to measure the performances etc.
    timer = None

    # Stores whether the trainer shall use wandb to sync dev data
    use_wandb = None

    # Stores the current data set (e.g. when multiple jobs use the same dataset)
    current_dataset = None

    # When true, trainer will output much more details about jobs progress
    debug = None

    # Attribute that stores the jobs that still need to be done
    job_queue = None

    def __init__(self, jobs=None, debug=False, wandb=False):
        """
        Constructor of trainer where some basic operations are done

        :param jobs: a list of json files to be executed
        :param debug: when debug mode is on, more status messages will be printed
        :param wandb: uses wandb when true and possible to sync dev information
        """

        # Print loading message
        print(bcolors.OKBLUE + "INFO: Trainer is setting up and initiating job queue  ..." + bcolors.ENDC)

        # Obtain the base path at looking at the parent of the parents parent
        base_path = Path(__file__).parent.parent.parent.resolve()

        # Create log path
        log_path = os.path.join(base_path, 'log')

        # Check if log dir exists, if not create
        Path(log_path).mkdir(parents=True, exist_ok=True)

        # Save whether debug and wandb shall be true or false
        self.debug = debug
        self.use_wandb = wandb

        # Create logger
        self.logger = Logger(log_path, file_name='log')

        # Create a timer instance for time measurements
        self.timer = Timer()

        # Create an instance of the model
        self.model = OrganNet25D()

        # Initialize the job queue
        self.job_queue = []

        # Iterate through the passed jobs
        for job in jobs:

            # Create the absolut path to the job file
            job_path = os.path.join(base_path, job)

            # Check whether the job file exist
            if os.path.exists(job_path):

                try:
                    # Open the file that contains the job description
                    f = open(job_path)

                    # Load the json data from the file
                    job_data = json.load(f)

                    # Check whether the job data is okay
                    if self._check_job_data(job_data):

                        # Append the job data to the job queue
                        self.job_queue.append(job_data)

                except Exception:
                    raise ValueError(bcolors.FAIL + "ERROR: Job file can not be read (" + job + ")" + bcolors.ENDC)

            else:
                print(bcolors.FAIL + "ERROR: Given job path does not exist (" + job + ")" + bcolors.ENDC)

    def run(self):
        """
        This method can be called in order to run a job (encoded in json format)
        """

        # Iterate through all jobs
        for index, job in enumerate(self.job_queue):

            # Print loading message
            print(bcolors.OKBLUE + "INFO: Started job " + str(index) + ": " + job['name'] + bcolors.ENDC)

            # Check if model shall be resetted
            if job['model']['reset']:

                # Create an instance of the model
                self.model = OrganNet25D()

                # Print loading message
                print(bcolors.OKBLUE + "INFO: Resetted OrganNet25D" + bcolors.ENDC)

            # Check if job contains index "training"
            if 'training' in job and type(job['training']) is dict:

                # Print loading message
                print(bcolors.OKBLUE + "INFO: Starting training of the model" + bcolors.ENDC)

                # Call train method
                self._train(job)

            # Check if job contains index "training"
            if 'evaluation' in job and type(job['evaluation']) is dict:
                # Print loading message
                print(bcolors.OKBLUE + "INFO: Starting evaluation of model" + bcolors.ENDC)

                # Call evaluation method
                self._evaluate(job)

            # Check if job contains index "training"
            if 'inference' in job and type(job['inference']) is dict:

                # Print loading message
                print(bcolors.FAIL + "ERROR: Inference is not implemented yet" + bcolors.ENDC)

    def _train(self, job: dict):
        """
        This method will train the network

        :param job: the dict containing everything regarding the current job
        """

        # Get dataset if not given
        dataset = Trainer.get_dataset()

        # Get dataloader for both training and validation
        train_data, val_data = Trainer.get_dataloader(dataset, split_ratio=self.split_ratio, num_workers=self.num_workers, batch_size=self.batch_size)

        # Log dataset information
        self.logger.write('{} samples.'.format(len(dataset)))
        best_val_loss = math.inf

        # --------------------------- Training routine --------------------------
        # Iterate through epochs
        for epoch in range(self.epochs):

            # Start epoch timer
            self.logger.write('Epoch {}/{}'.format(epoch + 1, self.epochs))
            self.timer.start('epoch')

            # Set model to train mode
            self.model.train()

            # Initialize variables
            running_loss = 0

            # ------------------------- Batch training -------------------------
            for batch, batch_input in enumerate(train_data):
                # Reset gradients
                self.optimizer.zero_grad()

                # Load data to device
                batch_input = batch_input.to(self.device)

                # Get output
                reconstructed = self.model(batch_input)

                # Calculate loss
                loss = self.loss_function(reconstructed, batch_input)

                # Backpropagation
                loss.backward()

                # Perform optimization step
                self.optimizer.step()

                # Add loss
                running_loss += loss.detach().cpu().numpy()

            # Calculate epoch los
            epoch_train_loss = running_loss / len(train_data)

    def _evaluate(self, job: dict):
        """
        This method will evaluate the network

        :param job: the dict containing everything regarding the current job
        """
        a = 0

    def _check_job_data(self, job_data: dict):
        """
        This method checks whether a passed job (in terms of a path to a json file) contains everything needed

        :param job_data: a dict that stores all job data
        :return: job data is okay and contains everything
        """

        # TODO: implement this tests later (prioritizing!)

        return True

    def get_dataset(self, data: dict):
        """
        Method creates the data set instance and returns it based on the data (contains job description)

        :return: CTDataset instance that contains samples
        """

        # TODO: check for self.current_dataset

        # Create an instance of the dataloader and pass location of data
        dataset = CTDataset('./data', preload=True)

        return dataset

    def get_dataloader(self, dataset, shuffle: bool = True, split_ratio: float = 0.5, num_workers: int = 0, batch_size: int = 64, pin_memory: bool = True):
        """
        The method returns data loader instances (if split) or just one dataloader based on the passed dataset

        :param dataset: the data set that the data loader should work on
        :param shuffle: whether the data shall be shuffled
        :param split_ratio: the ratio that the split shall be based on (if none, no split)
        :param num_workers: number of workers for laoding data
        :param batch_size: batch size of returned samples
        :param pin_memory:
        :return:
        """

        # Check whether the user wants a split data set
        if split_ratio is not None:

            # Determine split threshold and perform random split of the passed data set
            split_value = int(split_ratio * len(dataset))
            first_split, second_split = random_split(dataset,
                                                     [split_value, len(dataset) - split_value],
                                                     generator=torch.Generator().manual_seed(10))

            # Initialize data loaders for both parts of the split data set
            first_split = DataLoader(first_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
            second_split = DataLoader(second_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

            # Return tupel of splits
            return first_split, second_split

        else:

            # When no split is wanted, just return the data loader
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory), None
