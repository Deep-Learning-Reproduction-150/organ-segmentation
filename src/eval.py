"""
This file contains the evaluator class which evaluates the trained model

Course: Deep Learning
Date: 04.04.2022
Group: 150
"""

import torch

from src.utils import Logger
from src.losses import DiceCoefficient


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

    def evaluate(self, trained_model, evaluation_set, job_path, sample_freq=3):
        """
        Evaluate the model based on the dice score
        :return:
        """

        Logger.log("Creating some really cool metrics with the trained model <3", in_cli=True)

        # TODO: do some awesome evaluations of the model
        trained_model.eval()
        sum_of_avg = 0
        dsc_per_ch = []

        sample_inputs = []
        sample_labels = []
        sample_outputs = []

        i = 0
        for batch, batch_input in enumerate(evaluation_set):
            inputs, labels = batch_input

            b = inputs.shape[0]
            i += b
            pred = trained_model(inputs)

            avg_dsc, per_ch = DiceCoefficient()(pred, labels, return_per_channel_dsc=True)

            dsc_per_ch.append(per_ch)
            sum_of_avg += avg_dsc

            if batch % sample_freq == 0:  # Sample every fifth image
                sample_inputs.append(inputs)
                sample_labels.append(labels)
                sample_outputs.append(pred)

        dsc_per_ch = torch.stack(dsc_per_ch).mean(dim=0).detach().numpy().round(decimals=2)
        mean_dsc = sum_of_avg / len(evaluation_set)
        # Return something interesting

        evaluation_results = {
            "mean_dsc": mean_dsc,
            "dsc_per_ch": dsc_per_ch,
            "sample_inputs": torch.cat(sample_inputs),  # assume batch size 1
            "sample_labels": torch.cat(sample_labels),
            "sample_outputs": torch.cat(sample_outputs),
        }
        msg = f"""  
                Evaluation results:\n
                \t N images: \t {i}\n
                \t Average DSC:\t{mean_dsc}\n
                \t Per organ: \t {dsc_per_ch}
                """
        Logger.log(msg, type="INFO", in_cli=True)

        return evaluation_results
