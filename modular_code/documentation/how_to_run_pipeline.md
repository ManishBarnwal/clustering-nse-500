### How to run this pipeline of tasks?

**Make sure to initiate the luigi daemon by typing `luigid` in the terminal**

1. Find the task that is at lowest level - say NSECompaniesSegmentation task

2. In the terminal, go to the path where this file is located.

    a. `cd modular_code/training`

3. Run the luigi pipeline in local mode by typing
    `python -m luigi --module nse_companies_clustering NSECompaniesSegmentation --local-schedule`