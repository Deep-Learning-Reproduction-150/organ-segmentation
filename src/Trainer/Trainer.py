"""
This file contains the trainer class which performs jobs

Course: Deep Learning
Date: 03.04.2022
Group: 150
"""

import torch
import math
import importlib
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

                    # Append the job data to the job queue
                    self.job_queue.append(self._check_job_data(job_data))

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
                self._train(job['training'])

            # Check if job contains index "training"
            if 'evaluation' in job and type(job['evaluation']) is dict:
                # Print loading message
                print(bcolors.OKBLUE + "INFO: Starting evaluation of model" + bcolors.ENDC)

                # Call evaluation method
                self._evaluate(job['evaluation'])

            # Check if job contains index "training"
            if 'inference' in job and type(job['inference']) is dict:

                # Print loading message
                print(bcolors.FAIL + "ERROR: Inference is not implemented yet" + bcolors.ENDC)

    def _train(self, training_setup: dict):
        """
        This method will train the network

        :param training_setup: the dict containing everything regarding the current job
        """

        # Get dataset if not given
        dataset = self._get_dataset(training_setup['dataset'])

        # Get dataloader for both training and validation
        train_data, val_data = self._get_dataloader(dataset,
                                                   split_ratio=training_setup['split_ratio'],
                                                   num_workers=training_setup['num_workers'],
                                                   batch_size=training_setup['batch_size'])

        # Log dataset information
        self.logger.write('{} samples.'.format(len(dataset)))

        # Create a variable that stores the best value loss
        best_val_loss = math.inf

        # Create optimizer
        optimizer = self._get_optimizer(training_setup['optimizer'])

        # Create loss function
        loss_function = self._get_loss_function(training_setup['loss'])

        # --------------------------- Training routine --------------------------
        # Iterate through epochs
        for epoch in range(training_setup['epochs']):

            # Start epoch timer
            self.logger.write('Epoch {}/{}'.format(epoch + 1, training_setup['epochs']))
            self.timer.start('epoch')

            # Set model to train mode
            self.model.train()

            # Initialize variables
            running_loss = 0

            # ------------------------- Batch training -------------------------
            for batch, batch_input in enumerate(train_data):
                # Reset gradients
                optimizer.zero_grad()

                # Get output
                reconstructed = self.model(batch_input)

                # Calculate loss
                loss = loss_function(reconstructed, batch_input)

                # Backpropagation
                loss.backward()

                # Perform optimization step
                optimizer.step()

                # Add loss
                running_loss += loss.detach().cpu().numpy()

            # Calculate epoch los
            epoch_train_loss = running_loss / len(train_data)

    def _evaluate(self, evaluation_setup: dict):
        """
        This method will evaluate the network

        :param evaluation_setup: the dict containing everything regarding the current job
        """
        a = 0

    def _check_job_data(self, job_data: dict):
        """
        This method checks whether a passed job (in terms of a path to a json file) contains everything needed

        :param job_data: a dict that stores all job data
        :return: job data is okay and contains everything
        """

        # TODO: implement this tests and default autocomplete later (prioritizing!)

        # TODO: flash warnings when specific parts of the job description are missing and defaults are used

        return job_data

    def _get_optimizer(self, optimizer_setup: dict, **params):
        if optimizer_setup['name'] == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=optimizer_setup['learning_rate'], betas=optimizer_setup['betas'], **params)
        else:
            raise ValueError(bcolors.FAIL + "ERROR: Optimizer " + optimizer_setup['name'] + " not recognized, aborting" + bcolors.ENDC)

    def _get_loss_function(self, name: str, **params):
        module = importlib.import_module('src.losses')
        loss_class = getattr(module, name)
        return loss_class(**params)

    def _get_dataset(self, data: dict):
        """
        Method creates the data set instance and returns it based on the data (contains job description)

        :return: CTDataset instance that contains samples
        """

        # Obtain the base path at looking at the parent of the parents parent
        base_path = Path(__file__).parent.parent.parent.resolve()

        # Generate the path where the data set is located at
        dataset_path = os.path.join(base_path, data['root'])

        # Create an instance of the dataloader and pass location of data
        dataset = CTDataset(dataset_path, preload=data['preload'], transform=data['transform'])

        return dataset

    def _get_dataloader(self, dataset, shuffle: bool = True, split_ratio: float = 0.5, num_workers: int = 0, batch_size: int = 64, pin_memory: bool = True):
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
