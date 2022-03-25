"""
This main function enables one to call operation of this package via CLI

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

from src.OrganNet25D.network import OrganNet25D
from src.dataloader import DataLoader, ComputerTomographyImage


"""--------------------------- Part 1: Dataloading ------------------------------"""

# Create an instance of the dataloader and pass location of data
loader = DataLoader('./data', use_cross_validation=True)

# Get training and testing data sets from the data loader
training_data = loader.get_training_data()
testing_data = loader.get_testing_data()


"""-------------------------- Part 2: Model Training ----------------------------"""

# Create an instance of the OrganNet25D model
model = OrganNet25D()

# Train the model with the data sets (contains validation etc.)
model.train(train_data=training_data, test_data=testing_data, monitor_progress=True)


"""------------------------ Part 3: Model Inferencing ---------------------------"""

result = model.get_organ_segments(loader.get_random_example())
