"""
This file contains the runner class which runs jobs

Course: Deep Learning
Date: 03.04.2022
Group: 150
"""

import torch
import wandb
import importlib
import json
import os
from src.utils import Logger, Timer, bcolors
from pathlib import Path
from torch.utils.data import random_split, DataLoader
from src.Data.utils import CTDataCollator
from src.Model.OrganNet25D import OrganNet25D
from src.Data.CTDataset import CTDataset


class Runner:
    """
    This trainer instance performs the training of the network by executing jobs

    TODO:
        - feel free to add ideas what the trainer should be able to do
        - we should definitely include a "resume" option in the trainer, so save pytorch checkpoints (as .tar)
    """

    # Attribute stores an instance of the network
    model = None

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

        # Obtain the base path at looking at the parent of the parents parent
        base_path = Path(__file__).parent.parent.parent.resolve()

        # Save whether debug and wandb shall be true or false
        self.debug = debug
        self.use_wandb = wandb

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
                # Print loading message
                Logger.out("Given job path does not exist (" + job + ")", type="ERROR")

    def run(self):
        """
        This method can be called in order to run a job (encoded in json format)
        """

        # Iterate through all jobs
        for index, job in enumerate(self.job_queue):

            # Create logger and clear the current file
            Logger.initialize(log_name=job['name'])

            # Reset the log file
            Logger.clear()

            # Print CLI message
            Logger.log("Started '" + job['name'] + "'", "INFO", self.debug)

            # Check if model shall be resetted
            if job['model']['reset']:

                # Create an instance of the model
                self.model = OrganNet25D()

                # Print CLI message
                Logger.log("Resetted OrganNet25D", in_cli=self.debug)

            # Check if job contains index "training"
            if 'training' in job and type(job['training']) is dict:

                # Print CLI message
                Logger.log("Starting training of the model", in_cli=self.debug)

                # Call train method
                self._train(job['training'])

            # Check if job contains index "training"
            if 'evaluation' in job and type(job['evaluation']) is dict:

                # Print CLI message
                Logger.log("Starting evaluation of model", in_cli=self.debug)

                # Call evaluation method
                self._evaluate(job['evaluation'])

            # Check if job contains index "training"
            if 'inference' in job and type(job['inference']) is dict:

                # Print CLI message
                Logger.log("Inference is not implemented yet", "ERROR", self.debug)

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
                                                    batch_size=training_setup['batch_size'],
                                                    batch_dimensions=tuple(training_setup['batch_dimensions']))

        # Log dataset information
        Logger.log(str(len(dataset)) + ' samples have been '
                          + ('loaded (preloading active)' if training_setup['dataset']['preload']
                             else 'found (preloading inactive)'), type="INFO", in_cli=self.debug)

        # Create optimizer
        optimizer = self._get_optimizer(training_setup['optimizer'])

        # Create loss function
        loss_function = self._get_loss_function(training_setup['loss'])

        # Iterate through epochs (based on jobs setting)
        for epoch in range(training_setup['epochs']):

            # Start epoch timer and log the start of this epoch
            Logger.log('Starting to run Epoch {}/{}'.format(epoch + 1, training_setup['epochs']), in_cli=False)

            # Start the epoch timer
            self.timer.start('epoch')

            # Set model to train mode
            self.model.train()

            # Initialize variables
            running_loss = 0

            # Run through batches and perform model training
            for batch, batch_input in enumerate(train_data):

                # Reset gradients
                optimizer.zero_grad()

                # Get output
                model_output = self.model(batch_input)

                # Calculate loss
                loss = loss_function(model_output, batch_input)

                # Backpropagation
                loss.backward()

                # Perform optimization step
                optimizer.step()

                # Add loss
                running_loss += loss.detach().cpu().numpy()

                # Print epoch status bar
                Logger.print_status_bar(
                    done=(epoch + 1 / int(training_setup['epochs'])) * 100,
                    title="epoch " + str(epoch + 1) + "/" + str(training_setup['epochs']) + " progress"
                )

            # Stop timer to measure epoch length
            epoch_time = self.timer.get_time('epoch')

            # Calculate epoch los
            epoch_train_loss = running_loss / len(train_data)

            # Log the epoch success
            Logger.log('Took : ' + str(epoch_time) + ', loss is ' + str(epoch_train_loss), in_cli=self.debug)

            # TODO: perform syncing with wandb

            # TODO: perform saving of checkpoints in training (current model state)

    def _evaluate(self, evaluation_setup: dict):
        """
        This method will evaluate the network

        :param evaluation_setup: the dict containing everything regarding the current job
        """
        evaluator = self._get_evaluator(evaluation_setup)
        evaluator.evaluate(self.model, )

    def _check_job_data(self, job_data: dict):
        """
        This method checks whether a passed job (in terms of a path to a json file) contains everything needed

        :param job_data: a dict that stores all job data
        :return: job data is okay and contains everything
        """

        # TODO: implement this tests and default autocomplete later (prioritizing!)

        # TODO: flash warnings when specific parts of the job description are missing and defaults are used

        self.id = job_data.setdefault('wand_id', wandb.util.generate_id())

        return job_data

    def _get_evaluator(self, evaluation_setup: dict):
        module = importlib.import_module('src.eval')
        evaluater_class = getattr(module, evaluation_setup['evaluator'])
        return evaluater_class(evaluation_setup)

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

    def _get_dataloader(self, dataset, shuffle: bool = True, split_ratio: float = 0.5, num_workers: int = 0,
                        batch_size: int = 64, pin_memory: bool = True, batch_dimensions: tuple = (128, 128, 128)):
        """
        The method returns data loader instances (if split) or just one dataloader based on the passed dataset

        :param dataset: the data set that the data loader should work on
        :param shuffle: whether the data shall be shuffled
        :param split_ratio: the ratio that the split shall be based on (if none, no split)
        :param num_workers: number of workers for laoding data
        :param batch_size: batch size of returned samples
        :param batch_dimensions: the desired dimensions of one sample in a batch
        :param pin_memory: speeds up data loading on GPU
        :return:
        """

        # Initialize the second split (as it might be none)
        second_split = None

        # Generate the data collator
        collator = CTDataCollator(batch_dimensions=batch_dimensions)

        # Check whether the user wants a split data set
        if split_ratio is not None:

            # Determine split threshold and perform random split of the passed data set
            split_value = int(split_ratio * len(dataset))
            first_split, second_split = random_split(dataset,
                                                     [split_value, len(dataset) - split_value],
                                                     generator=torch.Generator().manual_seed(10))

            # Initialize data loaders for both parts of the split data set
            first_split = DataLoader(first_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                     pin_memory=pin_memory, collate_fn=collator)
            second_split = DataLoader(second_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                      pin_memory=pin_memory, collate_fn=collator)

        else:

            # Just return one data loader then
            first_split = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                     pin_memory=pin_memory, collate_fn=collator)

        # Return tupel of splits
        return first_split, second_split
