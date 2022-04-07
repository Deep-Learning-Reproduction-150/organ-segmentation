"""
This file contains the evaluator class which evaluates the trained model

Course: Deep Learning
Date: 04.04.2022
Group: 150
"""
import torch

from src.utils import Logger
from src.losses import CombinedLoss, DiceCoefficient
from sklearn.model_selection import KFold


class BaseEvaluator:
    """
    This is the base evaluator class that implements some general methods that every evaluator shall have
    """

    def evaluate(self, trained_model, evaluation_set, job_path):
        """
        This method is abstract and has to be implemented by

        :param trained_model:
        :param job_path:
        :param evaluation_set:
        :return: the score of evaluation
        """
        pass


class ChenEvaluator(BaseEvaluator):
    """
    This evaluator is based on the dice score
    """

    # Configuration
    k_folds = 5
    n_epochs = 2
    results = {}

    def evaluate(self, trained_model, evaluation_set, job):
        """
        Evaluate the model based on the dice score
        :return:
        """

        Logger.log("Creating some really cool metrics with the trained model <3", in_cli=True)

        loss_function = CombinedLoss(job["training"]["loss"]["alpha"], job["training"]["loss"]["input_dim"])

        # Try resetting weights
        for layer in trained_model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Define K-fold Cross Validator
        kfold = KFold(n_splits=self.k_folds, shuffle=True)

        # Define model and set to evaluation mode
        model = trained_model.eval()

        # K-fold cross validation
        for current_fold, fold in enumerate(kfold.split(evaluation_set)):

            eval_running_loss = 0

            # Initiate dice loss per organ and total
            organ_dice_losses = {}
            dice_loss_fn = DiceCoefficient()

            # Evaluation for current fold
            for batch, batch_input in enumerate(fold):

                with torch.no_grad():

                    # Get inputs and generate outputs
                    inputs, labels = batch_input
                    outputs = model(inputs)

                    # Calculate loss
                    eval_loss = self.loss_function.forward(inputs, outputs)

                    # Add to running validation loss
                    eval_running_loss += eval_loss.detach().cpu().numpy()

                    # Iterate through channels and compute dice losses for metric logging
                    for i, organ in enumerate(job['training']['dataset']['labels']):
                        sub_tensor = outputs[:, i, :, :, :]
                        sub_label = labels[:, i, :, :, :]
                        if organ not in organ_dice_losses.keys():
                            organ_dice_losses[organ] = []
                        organ_dice_losses[organ].append(float(dice_loss_fn(sub_tensor, sub_label)))
                    if "Background" not in organ_dice_losses.keys():
                        organ_dice_losses["Background"] = []
                    organ_dice_losses["Background"].append(
                        float(
                            dice_loss_fn(
                                outputs[:, len(job['training']['dataset']['labels']), :, :, :],
                                labels[:, len(job['training']['dataset']['labels']), :, :, :],
                            )
                        )
                    )

                    # Get the current running los
                    current_loss = eval_running_loss / batch if batch > 0 else eval_running_loss

                    # Print epoch status bar
                    Logger.print_status_bar(
                        done=((batch + 1) / len(evaluation_set)) * 100,
                        title="evaluation loss: " + "{:.5f}".format(current_loss)
                    )

                # Mean over the dice losses
                for key, val in organ_dice_losses.items():
                    organ_dice_losses[key] = sum(organ_dice_losses[key]) / len(organ_dice_losses[key])

                # Print epoch status bar
                Logger.print_status_bar(done=((batch + 1) / len(evaluation_set)) * 100, title="validating model")

                # Print loss for last fold
                Logger.log("Loss for fold" + current_fold + ": " + eval_loss)

                # End status bar
                Logger.end_status_bar()

        # Return epoch train val loss
        return eval_running_loss / len(evaluation_set)

