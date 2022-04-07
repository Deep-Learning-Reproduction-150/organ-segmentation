"""
This file contains the evaluator class which evaluates the trained model

Course: Deep Learning
Date: 04.04.2022
Group: 150
"""

from src.utils import Logger


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

    def evaluate(self, trained_model, evaluation_set, job_path):
        """
        Evaluate the model based on the dice score
        :return:
        """

        Logger.log("Creating some really cool metrics with the trained model <3", in_cli=True)

        # TODO: do some awesome evaluations of the model

        # Return something interesting
        return 100
