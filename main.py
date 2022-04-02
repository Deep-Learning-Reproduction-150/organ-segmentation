"""
This main function enables one to call operation of this package via CLI

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

from src.Trainer.Trainer import Trainer


# Gather the wanted jobs in a list (only testing job for now)
jobs = [
    'config/testing.json'
]

# Create a trainer object and call him robert
robert = Trainer(jobs=jobs, debug=True, wandb=False)

# Run the jobs
robert.run()
