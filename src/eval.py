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

    def evaluate(self, model, eval_set):
        """
        This method is abstract and has to be implemented by

        :param model: the model to evaluate
        :param eval_set: the data set for evaluation
        :return: the score of evaluation
        """
        pass


class DiceEvaluator(BaseEvaluator):
    """
    This evaluator is based on the dice score

    TODO: Some notes from the paper that we shall output
        - Dice Score
        - Confidence Interval
        - k-fold cross validation
    """

    def __init__(self, setup):
        a = 0

    def evaluate(self, model, eval_set):
        """
        Evaluate the model based on the dice score
        :return:
        """

        Logger.log("Creating some really cool metrics with the trained model <3", in_cli=True)

        # TODO: do some awesome evaluations of the model

        # Return something interesting
        return 100
