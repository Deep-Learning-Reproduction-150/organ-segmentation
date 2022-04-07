"""
A sample main function to run this program. You can also use the Runner
in your own context or pass it dicts to dynamically create configs.

Delft University of Technology
Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

from src.Runner.Runner import Runner


# Add all the jobs, that you want to run, here
jobs = [
    # "config/testing-7.json",
    "config/activation-testing-3-Cross.json",
    "config/activation-testing-3-MSE.json",
    "config/activation-testing-2-Cross.json",
    "config/activation-testing-2-MSE.json",
    "config/activation-testing-1-Cross.json",
    "config/activation-testing-1-MSE.json",
]

# Main guard for multithreading the runner "below"
if __name__ == "__main__":

    # Create a runner instance and pass it the jobs
    worker = Runner(jobs=jobs, debug=True)

    # Start working on the jobs until all are finished
    worker.run()
