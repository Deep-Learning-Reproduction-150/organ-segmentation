"""
This file contains the evaluator class which evaluates the trained model

Course: Deep Learning
Date: 04.04.2022
Group: 150
"""
import torch

from src.utils import Logger
from src.losses import CombinedLoss
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
    loss_function = CombinedLoss()
    results = {}

    def evaluate(self, trained_model, evaluation_set, job_path):
        """
        Evaluate the model based on the dice score
        :return:
        """

        Logger.log("Creating some really cool metrics with the trained model <3", in_cli=True)

        # Try resetting weights
        for layer in trained_model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Define K-fold Cross Validator
        kfold = KFold(n_splits=self.k_folds, shuffle=True)

        model = trained_model

        # K-fold cross validation
        for fold in enumerate(kfold.split(evaluation_set)):

            # Evaluation for current fold
            correct, total = 0, 0
            with torch.no_grad():

                # Iterate over validation data and generate predictions
                for i, data in enumerate(evaluation_set, 0):

                    # Get inputs and generate outputs
                    inputs, targets = data
                    outputs = model(inputs)

                    # Set correct and total
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum()#.item()

                print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
                Logger.log("Accuracy for fold" + fold + ": " + 100.0 * correct / total)

        return 100
