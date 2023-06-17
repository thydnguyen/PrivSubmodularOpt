This document contains the files to replicate the experiments for the Uber Pickups and the synthetic datasets.
The files contain:
-- The Uber Pickups dataset (uber-data-large.csv)
-- The synthetic dataset (synth.npz)
-- The code to run the Uber Pickups experiment (taxi_cab_grid.py)
-- The code to run the synthetic dataset experiment (synth_data_grid.py)
-- The bash file to run all the experiments included in the paper (runAll.sh)
-- The csv file of the result for all the experiments of the Uber pickups dataset (result_taxi_formatted.csv)
-- The csv file of the result for all the experiments of the synthetic dataset (result_synth_formatted.csv)

The taxi_cab_grid.py and synth_data_grid.py require the following Python packages: numpy, scipy, pandas. 
To run the scripts using the command-line, navigate to the folder and enter:

"python taxi_cab_grid.py (or synth_data_grid.py) --numIter [number of iterations for taking average] --gridsize [size of each grid dimension]
--theta [Theta paramter of the algorithm] --epsilon [Epsilon privacy paramter of the algorithm] --delta [Delta privacy paramter of the algorithm] --K [number of centers as a fraction of the dataset] --seed [random seed default to 2022] --overwrite (if set, overwrite the output file, otherwise append to it)
--random ( run the Random approach) --laplace (run the Laplace noise approach) --gumbel (run the gumbel noise approach) --non-private (run the non-private approach)
--batch (run the non-private non-streaming approach) --batchpriv (run the private non-streaming approach)

The python file will print out the mean and standard deviation of each of the approach (random, laplace, gumbel, non-private respectively) on the screen. The result will be added to the csv files (result_synth_formatted.csv or result_taxi_formatted.csv) along with values of K, Epsilon, and Theta. Values for methods not specified in the command-line will be recorded as 'nan'. 


