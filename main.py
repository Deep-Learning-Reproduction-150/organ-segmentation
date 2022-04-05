"""
This main function enables one to call operation of this package via CLI

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

from src.Runner.Runner import Runner


# Gather the wanted jobs in a list (only testing job for now)
jobs = [
    #'config/testing.json',
    'config/paper_reproduction.json',
]

# Create a trainer object and call him robert
worker = Runner(jobs=jobs, debug=True)

# Run the jobs
worker.run()
