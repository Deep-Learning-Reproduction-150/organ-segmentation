"""
This main function enables one to call operation of this package via CLI

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

from src.OrganNet25D.network import OrganNet25D
from src.Dataloader.CTDataset import CTDataset


"""--------------------------- Part 1: Dataloading ------------------------------"""

# Create an instance of the dataloader and pass location of data
dataset = CTDataset('./data', use_cross_validation=True)

# Create a GIF that shows every single data sample (TODO: comment out after you have them!)
# dataset.create_all_visualizations(direction='vertical')

# Visualize a random sample from the data
random_sample = dataset.__getitem__(5)
random_sample.visualize(export_gif=True, high_quality=True, export_png=True, direction='horizontal')

"""-------------------------- Part 2: Model Training ----------------------------"""

# Create an instance of the OrganNet25D model
model = OrganNet25D()

# Train the model with the data sets (contains validation etc.)
# TODO: do this in PyTorch logic

"""------------------------ Part 3: Model Inferencing ---------------------------"""

result = model.get_organ_segments(dataset.__getitem__(2))
