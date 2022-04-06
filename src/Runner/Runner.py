"""
This file contains the runner class which runs jobs

Course: Deep Learning
Date: 03.04.2022
Group: 150
"""

import torch
import importlib
import random
import json
import wandb
import os
import numpy as np
from src.utils import Logger, Timer, bcolors
from src.losses import DiceCoefficient
from pathlib import Path
from torch.utils.data import random_split, DataLoader
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

    # The path to the current jobs directory (in ./logs/<job_name>)
    path = None

    # Attribute will contain the last checkpoint if exists (else none)
    checkpoint = None

    # An instance of a timer to measure the performances etc.
    timer = None

    # A data set that will overwrite the data set specified in the job
    dataset = None

    # The current train data set that is used by the runner
    train_data = None

    # The current evaluation data set that is used by the runner
    eval_data = None

    # When true, trainer will output much more details about jobs progress
    debug = None

    # This attribute stores the current job the runner is working on
    job = None

    # Attribute that stores the jobs that still need to be done
    job_queue = None

    def __init__(self, jobs=None, debug=False, dataset=None):
        """
        Constructor of trainer where some basic operations are done

        :param jobs: a list of json files to be executed
        :param debug: when debug mode is on, more status messages will be printed
        :param dataset: A data set that will overwrite the data set specified in the job

        TODO: OVERWRITE dataset if you pass a dataset
        """

        # Obtain the base path at looking at the parent of the parents parent
        base_path = Path(__file__).parent.parent.parent.resolve()

        # Save whether debug and wandb shall be true or false
        self.debug = debug

        # A data set that will overwrite the data set specified in the job
        self.dataset = dataset

        # Initialize eval and train data variables
        self.eval_data = None
        self.train_data = None

        # Create a timer instance for time measurements
        self.timer = Timer()

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
                print(bcolors.FAIL + "Given job path does not exist (" + job + ")" + bcolors.ENDC)

    def run(self):
        """
        This method can be called in order to run a job (encoded in json format)
        """

        # Iterate through all jobs
        for index, job in enumerate(self.job_queue):

            # Save the current job for instance access
            self.job = job

            # Obtain the base path at looking at the parent of the parents parent
            base_path = Path(__file__).parent.parent.parent.resolve()

            # A directory that stores jobs data
            job_data_dir = os.path.join(base_path, "jobs", job["name"])

            # Save this path to the runner object for now to be able to store stuff in there
            self.path = job_data_dir

            # Check if log dir exists, if not create
            Path(job_data_dir).mkdir(parents=True, exist_ok=True)

            # Create logger and clear the current file
            Logger.initialize(log_path=job_data_dir)

            # Reset the log file
            if not job["resume"]:
                Logger.clear()

            # Print CLI message
            Logger.log("Started the job '" + job["name"] + "'", "HEADLINE", self.debug)

            # check whether the job description has changed (if that is the case, re-run the job)
            specification_path = os.path.join(self.path, "specification.json")

            # if os.path.exists(specification_path):
            #     existing_specification = json.load(open(specification_path))
            #     if str(existing_specification) != str(job):
            #
            #         # Remove the last checkpoint
            #         checkpoint_path = os.path.join(self.path, 'checkpoint.tar')
            #         if os.path.exists(checkpoint_path):
            #             os.remove(checkpoint_path)
            #
            #         # Notify user regarding rerunning of job
            #         Logger.log("The json job specification has changed, deleting checkpoint", type="WARNING",
            #         in_cli=True)

            # Write the specification file to the job
            with open(specification_path, "w") as fp:
                json.dump(job, fp)

            # Create an instance of the model TODO: could be passing different models here? Via job.json?
            self.model = OrganNet25D(
                input_shape=job["model"]["input_shape"], hdc_dilations=job["model"]["hdc_dilations"]
            )

            # Recover the last checkpoint (if exists)
            if job["resume"]:

                # Load the last checkpoint
                self.checkpoint = self._load_checkpoint()

                # Check if checkpoint exists
                if self.checkpoint is not None:

                    # Recover model from last
                    self.model.load_state_dict(self.checkpoint["model"])

                    # Log a status message about recovery of model
                    Logger.log("Recovered model from the last checkpoint", type="WARNING", in_cli=True)

            else:
                self.checkpoint = None

            # Check if job contains index "training"
            if "training" in job and type(job["training"]) is dict:

                # Call train method
                self._train()

            # Check if job contains index "training"
            if "evaluation" in job and type(job["evaluation"]) is dict:

                # Call evaluation method
                self._evaluate(job["evaluation"])

            # Check if job contains index "training"
            if "inference" in job and type(job["inference"]) is dict:

                # Print CLI message
                Logger.log("Inference is not implemented yet", "ERROR", self.debug)

    def _train(self):
        """
        This method will train the network

        :param training_setup: the dict containing everything regarding the current job
        """

        # Check if the model has been trained in a previous run already
        if self.checkpoint is not None:

            # Check if training is already done
            if self.checkpoint["training_done"]:
                # Log that training has already been done
                Logger.log("Model is already fully trained, skipping training", type="SUCCESS", in_cli=True)
                return True

            # Extract epoch to continue training
            start_epoch = self.checkpoint["epoch"] + 1
            wandb_id = self.checkpoint["wandb_run_id"]

        else:

            # Fallback to a default start epoch of zero
            start_epoch = 0
            wandb_id = None

        # Check if wandb shall be used
        if self.job["wandb_api_key"]:

            # Disable wandb console output
            os.environ["WANDB_SILENT"] = "true"
            wandb.login(key=self.job["wandb_api_key"])
            if start_epoch > 0 and wandb_id is not None:
                # Flash notification
                Logger.log("Loading wand and attempting to resume run " + str(wandb_id))
                self.wandb_worker = wandb.init(id=wandb_id, project=self.job["wandb_project_name"], resume='allow')
            else:
                # Flash notification
                Logger.log("Loading wand for project " + self.job["wandb_project_name"])
                self.wandb_worker = wandb.init(project=self.job["wandb_project_name"])

        # Start timer to measure data set
        self.timer.start("creating dataset")

        # Get dataset if not given
        dataset = self._get_dataset(self.job["training"]["dataset"], preload=self.job["preload"])

        # Start timer to measure data set
        creation_took = self.timer.get_time("creating dataset")

        # This variable eventually contains dice scores that are created in evaluation
        organ_dice_losses = {}

        # Notify about data set creation
        dataset_constructor_took = "{:.2f}".format(creation_took)
        Logger.log("Generation of data set took " + dataset_constructor_took + " seconds", in_cli=True)

        # Get dataloader for both training and validation
        self.train_data, self.eval_data = self._get_dataloader(
            dataset,
            split_ratio=self.job["training"]["split_ratio"],
            num_workers=self.job["training"]['dataset']["num_workers"],
            batch_size=self.job["training"]["batch_size"],
        )

        # Log dataset information
        Logger.log(
            "Start training on "
            + str(len(self.train_data))
            + " batches "
            + ("(preloading active)" if self.job["preload"] else "found (preloading inactive)"),
            type="INFO",
            in_cli=self.debug,
        )

        # Create optimizer
        optimizer = self._get_optimizer(self.job["training"]["optimizer"])

        # Create scheduler
        scheduler = self._get_lr_scheduler(optimizer, self.job["training"]["lr_scheduler"])

        # Create loss function
        loss_function = self._get_loss_function(self.job["training"]["loss"])

        # Enable wandb logging
        if self.job["wandb_api_key"]:
            self.wandb_worker.watch(self.model, log="gradients", log_freq=1)

        # Check if start epoch is not zero and notify
        if start_epoch > 0:

            # Print notification
            Logger.log("Resuming training in epoch " + str(start_epoch), in_cli=True)

        if self.job["training"]["detect_bad_gradients"]:
            Logger.log("Selected detect_bad_gradients - using AutoGrad", type="WARNING", in_cli=True)

        # Iterate through epochs (based on jobs setting)
        for epoch in range(start_epoch, self.job["training"]["epochs"]):

            # Start epoch timer and log the start of this epoch
            Logger.log("Starting to run Epoch {}/{}".format(epoch + 1, self.job["training"]["epochs"]), in_cli=True, new_line=True)

            # Print epoch status bar
            Logger.print_status_bar(done=0, title="epoch " + str(epoch + 1) + "/" + str(self.job["training"]["epochs"]))

            # Initiate a model output, input and labels variable with none
            model_output = None
            labels = None
            inputs = None

            # Start the epoch timer
            self.timer.start("epoch")

            # Set model to train mode
            self.model.train()

            # Initialize variables
            running_loss = 0

            # Run through batches and perform model training
            for batch, batch_input in enumerate(self.train_data):

                # Extract inputs and labels from the batch input
                inputs, labels = batch_input

                # Reset gradients
                optimizer.zero_grad()

                # Get output
                model_output = self.model(inputs)

                # Calculate loss TODO: the labels are int8 to save storage
                loss = loss_function(model_output, labels.to(torch.float32))

                if self.job["training"]["detect_bad_gradients"]:
                    from torch import autograd

                    with autograd.detect_anomaly():
                        loss.backward()
                else:
                    # Backpropagation
                    loss.backward()

                # Gradient clipping
                # grad_norm = torch.nn.utils.clip_grad_norm_(
                #     self.model.parameters(), self.job["training"]["grad_norm_clip"]
                # )

                # Perform optimization step
                optimizer.step()

                # Add loss
                running_loss += loss.detach().cpu().numpy()

                # Get the current running los
                current_loss = running_loss / batch if batch > 0 else running_loss

                # Print epoch status bar
                Logger.print_status_bar(
                    done=((batch + 1) / len(self.train_data)) * 100,
                    title="epoch "
                    + str(epoch + 1)
                    + "/"
                    + str(self.job["training"]["epochs"])
                    + ", loss: "
                    + "{:.2f}".format(current_loss),
                )

            # Finish the status bar
            Logger.end_status_bar()

            # Calculate epoch los
            epoch_train_loss = running_loss / len(self.train_data)

            # Perform validation
            if self.eval_data is not None:

                # Notify the user regarding validation
                Logger.log("Validating the model on " + str(len(self.eval_data)) + " validation batches ...")

                # Set model to evaluation mode
                self.model.eval()

                # Initialize a running loss of 99999
                eval_running_loss = 0

                # Initiate dice loss per organ and total
                organ_dice_losses = {}
                dice_loss_fn = DiceCoefficient()

                # Perform validation on healthy images
                for batch, batch_input in enumerate(self.eval_data):

                    # TODO: if using GPU, one could load the batch to the GPU now

                    # Extract inputs and labels from the batch input
                    inputs, labels = batch_input

                    # Calculate output
                    model_output = self.model(inputs)

                    # Determine loss TODO: the labels are int8 to save storage
                    eval_loss = loss_function(model_output, labels.to(torch.float32))

                    # Add to running validation loss
                    eval_running_loss += eval_loss.detach().cpu().numpy()

                    # Iterate through channels and compute dice losses for metric logging
                    for i, organ in enumerate(self.job['training']['dataset']['labels']):
                        sub_tensor = model_output[:, i, :, :, :]
                        sub_label = labels[:, i, :, :, :]
                        if organ not in organ_dice_losses.keys():
                            organ_dice_losses[organ] = []
                        organ_dice_losses[organ].append(float(dice_loss_fn(sub_tensor, sub_label)))
                    if "Background" not in organ_dice_losses.keys():
                        organ_dice_losses["Background"] = []
                    organ_dice_losses["Background"].append(
                        float(
                            dice_loss_fn(
                                model_output[:, len(self.job['training']['dataset']['labels']), :, :, :],
                                labels[:, len(self.job['training']['dataset']['labels']), :, :, :],
                            )
                        )
                    )

                    # Print epoch status bar
                    Logger.print_status_bar(done=((batch + 1) / len(self.eval_data)) * 100, title="validating model")

                # Mean over the dice losses
                for key, val in organ_dice_losses.items():
                    organ_dice_losses[key] = sum(organ_dice_losses[key]) / len(organ_dice_losses[key])

                # End status bar
                Logger.end_status_bar()

                # Calculate epoch train val loss
                epoch_evaluation_loss = eval_running_loss / len(self.eval_data)

                # TODO: here, we could do an early stopping if the model is extremely overfitting or so

            else:
                # If no validation is done, we take the train loss as val loss
                epoch_evaluation_loss = epoch_train_loss

            # Also perform a step for the learning rate scheduler
            scheduler.step()

            # Obtain the current learning rate
            current_lr = scheduler.get_last_lr()[0]

            # Print the current learning rate
            lr_formatted = "{:.6f}".format(current_lr)
            Logger.log("Learning rate currently at " + str(lr_formatted), in_cli=True)

            # Stop timer to measure epoch length
            epoch_time = self.timer.get_time("epoch")

            # Log the epoch success
            avg_loss = "{:.4f}".format(epoch_train_loss)
            avg_val_loss = "{:.4f}".format(epoch_evaluation_loss)
            Logger.log("Epoch took " + str(epoch_time) + " seconds. Training loss is " + avg_loss +
                       ", validation loss is " + avg_val_loss, in_cli=self.debug)

            # Report the current loss to wandb if it's set
            if self.job["wandb_api_key"]:

                # Log some prediction examples (with image overlays)
                self._log_prediction_examples(inputs, labels, model_output)
                self._log_prediction_examples_3d(inputs, labels, model_output)

                # Include max and min of predictions per organ
                self._log_prediction_max_min(model_output)

                # Log this current status
                self.wandb_worker.log(
                    {
                        "training loss": epoch_train_loss,
                        "evaluation loss": epoch_evaluation_loss,
                        "learning rate": current_lr,
                        "epoch duration": epoch_time,
                        "epoch": epoch + 1,
                        "DSC per channel": organ_dice_losses,
                    },
                    commit=False,
                )

                # Log all to wandb
                self.wandb_worker.log({})

            # Save a checkpoint for this job after each epoch (to be able to resume)
            self._save_checkpoint(
                {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "train-loss": epoch_train_loss,
                    "eval_loss": epoch_evaluation_loss,
                    "training_done": epoch == (self.job["training"]["epochs"] - 1),
                    "wandb_run_id": self.job["wandb_run_id"],
                }
            )

            # Write log message that the training has been completed
            Logger.log("Checkpoint updated for epoch " + str(epoch + 1) + " (data has been saved)", in_cli=True)

        # Write log message that the training has been completed
        Logger.log("Training of the model completed", type="SUCCESS", in_cli=True)

        # Check if wandb shall be used
        if self.job["wandb_api_key"]:
            self.wandb_worker.finish()

    def _evaluate(self, evaluation_setup: dict):
        """
        This method will evaluate the network

        :param evaluation_setup: the dict containing everything regarding the current job
        """

        # Log dataset information
        Logger.log("Start evaluation of the model", type="INFO", in_cli=self.debug)

        # Get the evaluation instance
        evaluator = self._get_evaluator(evaluation_setup)

        # Call evaluate on the evaluator
        evaluator.evaluate(self.model, self.eval_data, self.path)

        # Write log message that the training has been completed
        Logger.log("Evaluation of the model completed", type="SUCCESS", in_cli=True)

    def _check_job_data(self, job_data: dict):
        """
        This method checks whether a passed job (in terms of a path to a json file) contains everything needed

        :param job_data: a dict that stores all job data
        :return: job data is okay and contains everything
        """

        # TODO: implement this tests and default autocomplete later (prioritizing!)

        # TODO: flash warnings when specific parts of the job description are missing and defaults are used

        # Add a default scheduler
        job_data["training"].setdefault(
            "lr_scheduler",
            {"name": "LinearLR", "start_factor": 1, "end_factor": 0.01, "total_iters": 100},
        )

        # Set labels to none by default so the data set figures out the order
        job_data["training"]["dataset"].setdefault("labels", None)

        # Set a default if predictino examples is not set
        job_data.setdefault("wandb_prediction_examples", 6)

        # Append a wandb id if none exists yet
        job_data.setdefault("wandb_run_id", wandb.util.generate_id())

        return job_data

    def _get_evaluator(self, evaluation_setup: dict):
        """
        This method will return an evaluator that is specified in the jobs json file

        :param evaluation_setup:
        :return: evaluator instance
        """
        module = importlib.import_module("src.eval")
        evaluater_class = getattr(module, evaluation_setup["name"])
        return evaluater_class()

    def _log_prediction_examples_3d(self, inputs, labels, model_output):
        """
        Method logs prediction examples to wandb

        :param inputs:
        :param labels:
        :param model_output:
        """

        # Generate a random slice as example for reconstruction
        if model_output is not None:

            # Create a list of masks
            mask_list = []

            # Do 5 random slices to show what is happening
            for perspective_idx in range(0, 3):  # Loop through the spatial dims

                # Slice a random image from there
                batch_no = random.randint(0, inputs.size()[0] - 1)

                # Obtain the actual image
                sample_image = inputs[batch_no, 0, :, :, :].mean(dim=perspective_idx)

                # Create raw prediction and label masks
                prediction_mask_data = torch.ones_like(sample_image) * len(self.job['training']['dataset']['labels']) + 1
                label_mask_data = torch.ones_like(sample_image) * len(self.job['training']['dataset']['labels'])

                # Iterate through all organs and add them to it
                for organ_slice, organ in enumerate(self.job['training']['dataset']['labels']):
                    raw_prediction = model_output[batch_no, organ_slice, :, :, :].max(dim=perspective_idx).values
                    raw_label = labels[batch_no, organ_slice, :, :, :].max(dim=perspective_idx).values

                    prediction_mask_data = torch.where(
                        raw_prediction > 0.5, torch.tensor(organ_slice, dtype=torch.float32), prediction_mask_data
                    )
                    label_mask_data = torch.where(
                        raw_label > 0.5, torch.tensor(organ_slice, dtype=torch.float32), label_mask_data
                    )

                # Do the same for the background
                background_prediction = (
                    model_output[batch_no, len(self.job['training']['dataset']['labels']), :, :, :].max(dim=perspective_idx).values
                )
                prediction_mask_data = torch.where(
                    background_prediction > 0.5,
                    torch.tensor(len(self.job['training']['dataset']['labels']), dtype=torch.float32),
                    prediction_mask_data,
                )

                # Prepare class labels
                class_labels = {
                    len(self.job['training']['dataset']['labels']): "Background",
                    len(self.job['training']['dataset']['labels']) + 1: "No Prediction",
                }
                for i, organ in enumerate(self.job['training']['dataset']['labels']):
                    class_labels[i] = organ

                # Convert to ndarray for wandb
                input_image = sample_image.cpu().detach().numpy()

                # Append this slice to the predictions
                mask_list.append(
                    wandb.Image(
                        input_image,
                        caption="Perspective " + str(perspective_idx),
                        masks={
                            "predictions": {
                                "class_labels": class_labels,
                                "mask_data": prediction_mask_data.cpu().detach().numpy(),
                            },
                            "ground_truth": {
                                "class_labels": class_labels,
                                "mask_data": label_mask_data.cpu().detach().numpy(),
                            },
                        },
                    )
                )

            # Log all organ predictions
            self.wandb_worker.log(
                {
                    "predictions_3d": mask_list,
                },
                commit=False,
            )

        else:

            # Warn that there was no model output
            Logger.log("No prediction examples could be logged, as there is no model output", in_cli=True)

    def _log_prediction_examples(self, inputs, labels, model_output):
        """
        Method logs prediction examples to wandb

        :param inputs:
        :param labels:
        :param model_output:

        TODO: maybe there could be a smart way to actually find the most interesting slides?
        """

        # Generate a random slice as example for reconstruction
        if model_output is not None:

            # Start timer and log preperation operation
            self.timer.start("prediction-preperation")

            # Select a random sample from the batch
            batch_no = random.randint(0, inputs.size()[0] - 1)

            # Iterate through the model output and find good and bad slices
            good_slices_data = {"good": [], "bad": []}
            for s in range(labels.size()[-3] - 1):
                current_max = labels[batch_no, list(range(len(self.job['training']['dataset']['labels']) - 1)), s, :, :].max()
                if current_max > 0.5:
                    good_slices_data["good"].append(s)
                else:
                    good_slices_data["bad"].append(s)

            # Compose the good slices in a list
            if len(good_slices_data["good"]) > 0:
                first_good_slice = sorted(good_slices_data["good"])[0]
                last_good_slice = sorted(good_slices_data["good"])[-1]
                have_want_difference = (last_good_slice - first_good_slice + 1) - self.job["wandb_prediction_examples"]
                if have_want_difference >= 0:
                    good_slices = np.linspace(
                        first_good_slice, last_good_slice, num=self.job["wandb_prediction_examples"], dtype=int
                    )
                else:
                    possible_slices = self.job["wandb_prediction_examples"]
                    if self.job["wandb_prediction_examples"] > inputs.size()[-3]:
                        Logger.log(
                            "Requested "
                            + str(self.job["wandb_prediction_examples"])
                            + " slices, but can only return "
                            + str(inputs.size()[-3]),
                            type="ERROR",
                            in_cli=True,
                        )
                        possible_slices = inputs.size()[-3]
                    good_slices = sorted(good_slices_data["good"])
                    while len(good_slices) < possible_slices:
                        random_slice = random.randint(0, inputs.size()[-3] - 1)
                        if random_slice not in good_slices:
                            good_slices.append(random_slice)
                    good_slices = sorted(good_slices)
            else:
                good_slices = []
                for i in range(self.job["wandb_prediction_examples"]):
                    good_slices.append(random.randint(0, inputs.size()[-3] - 1))

            # Create a list of masks
            mask_list = []

            # Do 5 random slices to show what is happening
            for slice_no in good_slices:

                # Obtain the actual image
                sample_image = inputs[batch_no, 0, slice_no, :, :]

                # Create raw prediction and label masks
                prediction_mask_data = torch.ones_like(sample_image) * len(self.job['training']['dataset']['labels']) + 1
                label_mask_data = torch.ones_like(sample_image) * len(self.job['training']['dataset']['labels'])

                # Iterate through all organs and add them to it
                for organ_slice, organ in enumerate(self.job['training']['dataset']['labels']):
                    raw_prediction = model_output[batch_no, organ_slice, slice_no, :, :]
                    raw_label = labels[batch_no, organ_slice, slice_no, :, :]

                    # Create a dynamic threshold based on the median
                    dynamic_predict_threshold = float(
                        raw_prediction.min() + ((raw_prediction.max() - raw_prediction.min()) / 2)
                    )

                    prediction_mask_data = torch.where(
                        raw_prediction > dynamic_predict_threshold,
                        torch.tensor(organ_slice, dtype=torch.float32),
                        prediction_mask_data,
                    )
                    label_mask_data = torch.where(
                        raw_label > 0.5, torch.tensor(organ_slice, dtype=torch.float32), label_mask_data
                    )

                # Do the same for the background
                background_prediction = model_output[batch_no, len(self.job['training']['dataset']['labels']), slice_no, :, :]
                prediction_mask_data = torch.where(
                    background_prediction > float(background_prediction.median()),
                    torch.tensor(len(self.job['training']['dataset']['labels']), dtype=torch.float32),
                    prediction_mask_data,
                )

                # Prepare class labels
                class_labels = {
                    len(self.job['training']['dataset']['labels']): "Background",
                    len(self.job['training']['dataset']['labels']) + 1: "No Prediction",
                }
                for i, organ in enumerate(self.job['training']['dataset']['labels']):
                    class_labels[i] = organ

                # Convert to ndarray for wandb
                input_image = sample_image.cpu().detach().numpy()

                # Append this slice to the predictions
                mask_list.append(
                    wandb.Image(
                        input_image,
                        caption="Slice " + str(slice_no),
                        masks={
                            "predictions": {
                                "class_labels": class_labels,
                                "mask_data": prediction_mask_data.cpu().detach().numpy(),
                            },
                            "ground_truth": {
                                "class_labels": class_labels,
                                "mask_data": label_mask_data.cpu().detach().numpy(),
                            },
                        },
                    )
                )

            # Log all organ predictions
            self.wandb_worker.log(
                {
                    "predictions": mask_list,
                },
                commit=False,
            )

            # Notify about duration
            prep_took = self.timer.get_time("prediction-preperation")
            Logger.log("Preparing the examples took {:.2f} seconds".format(prep_took), in_cli=True)

        else:

            # Warn that there was no model output
            Logger.log("No prediction examples could be logged, as there is no model output", in_cli=True)

    def _log_prediction_max_min(self, model_output):

        max_vals = {}
        min_vals = {}
        for i, organ in enumerate(self.job['training']['dataset']['labels']):
            max_vals[organ] = model_output[:, i, :, :, :].max()
            min_vals[organ] = model_output[:, i, :, :, :].min()

        self.wandb_worker.log(
            {"predictions minimum value": min_vals, "predictions maximum value": max_vals}, commit=False
        )

    def _get_optimizer(self, optimizer_setup: dict, **params):
        if optimizer_setup["name"] == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=optimizer_setup["learning_rate"], betas=optimizer_setup["betas"], **params
            )
            if self.checkpoint is not None:
                Logger.log("Recovering optimizer from the last checkpoint", type="WARNING", in_cli=True)
                try:
                    optimizer.load_state_dict(self.checkpoint["optimizer"])
                except Exception:
                    Logger.log("Could not recover optimizer checkpoint state", type="ERROR", in_cli=True)
            return optimizer
        else:
            raise ValueError(
                bcolors.FAIL
                + "ERROR: Optimizer "
                + optimizer_setup["name"]
                + " not recognized, aborting"
                + bcolors.ENDC
            )

    def _get_lr_scheduler(self, optimizer, scheduler_setup: dict):
        """
        Returns a scheduler for the learning rate

        :param optimizer:
        :param scheduler_setup:
        :return:

        TODO: make this dynamic
        """

        # Every scheduler will need the optimizer
        scheduler_setup['optimizer'] = optimizer
        scheduler_name = scheduler_setup.pop('name')
        module = importlib.import_module("torch.optim.lr_scheduler")
        lr_scheduler_class = getattr(module, scheduler_name)
        scheduler = lr_scheduler_class(**scheduler_setup)

        # Check if there is a checkpoint
        if self.checkpoint is not None:
            Logger.log("Recovering scheduler from the last checkpoint", type="WARNING", in_cli=True)

            # Load the state dict
            try:
                scheduler.load_state_dict(self.checkpoint["lr_scheduler"])
            except Exception:
                Logger.log("Could not recover scheduler checkpoint state", type="ERROR", in_cli=True)

        # Return the scheduler
        return scheduler

    def _get_loss_function(self, loss_function_setup):
        module = importlib.import_module("src.losses")
        loss_class = getattr(module, loss_function_setup["name"])
        return loss_class(**loss_function_setup)

    def _get_dataset(self, data: dict, preload: bool = True):
        """
        Method creates the data set instance and returns it based on the data (contains job description)

        :return: CTDataset instance that contains samples
        """

        # Check if there is a passed data set that shall overwrite this
        if self.dataset is not None:

            # Warn user about overwriting the dataset
            Logger.log(
                "Dataset has been passed to runner. This overwrites the specification in the job config.",
                type="WARNING",
                in_cli=True,
            )

            # Return the overwrite data set
            return self.dataset

        # Obtain the base path at looking at the parent of the parents parent
        base_path = Path(__file__).parent.parent.parent.resolve()

        # Generate the path where the data set is located at
        dataset_path = os.path.join(base_path, data["root"])

        # Save a global version of the label order
        if data["labels"] is None:
            # Abort as the label structure is missing
            raise ValueError("You have to add the desired label structure to your job configuration (add a list at training/dataset/labels")

        # Create an instance of the dataloader and pass location of data
        dataset = CTDataset(
            dataset_path,
            preload=preload,
            label_transforms=data["label_transforms"],
            sample_transforms=data["sample_transforms"],
            label_structure=data["labels"],
            no_logging=False,
        )

        return dataset

    def _get_dataloader(
        self,
        dataset,
        shuffle: bool = True,
        split_ratio: float = 0.5,
        num_workers: int = 0,
        batch_size: int = 64,
        pin_memory: bool = False,
    ):
        """
        The method returns data loader instances (if split) or just one dataloader based on the passed dataset

        :param dataset: the data set that the data loader should work on
        :param shuffle: whether the data shall be shuffled
        :param split_ratio: the ratio that the split shall be based on (if none, no split)
        :param num_workers: number of workers for laoding data
        :param batch_size: batch size of returned samples
        :param pin_memory: speeds up data loading on GPU
        :return:
        """

        # Initialize the second split (as it might be none)
        second_split = None

        # Check whether the user wants a split data set
        if split_ratio is not None:

            # Determine split threshold and perform random split of the passed data set
            split_value = int(split_ratio * len(dataset))
            first_split, second_split = random_split(
                dataset, [split_value, len(dataset) - split_value], generator=torch.Generator().manual_seed(10)
            )

            # Initialize data loaders for both parts of the split data set
            first_split = DataLoader(
                first_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
            )
            second_split = DataLoader(
                second_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
            )

        else:

            # Just return one data loader then
            first_split = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
            )

        # Return tupel of splits
        return first_split, second_split

    def _load_checkpoint(self):
        """
        Returns the checkpoint dict found in path.

        TODO: map location could be switched to possible GPUs theoretically
        """

        # Generate a variable that stores the checkpoint path
        checkpoint_path = os.path.join(self.path, "checkpoint.tar")

        # Check if the file exists
        if not os.path.exists(checkpoint_path):
            return None

        # Load the checkpoint
        return torch.load(checkpoint_path, map_location=torch.device("cpu"))

    def _save_checkpoint(self, checkpoint_dict):
        """
        Saves a checkpoint dictionary in a tar object to load in case this job is repeated
        """
        save_path = os.path.join("results", self.path, "checkpoint.tar")
        torch.save(checkpoint_dict, save_path)

        # Check if wandb shall be used
        if self.job["wandb_api_key"]:
            wandb.save(save_path)
