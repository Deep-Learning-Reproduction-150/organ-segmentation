"""
This main function enables one to call operation of this package via CLI

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

from src.Runner.Runner import Runner




# Main guard
if __name__ == '__main__':

    # Create a trainer object and call him robert
    worker = Runner(jobs=jobs, debug=True)

    # Run the jobs
    worker.run()
