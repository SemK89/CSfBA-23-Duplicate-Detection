# CSfBA-23-Duplicate-Detection
Near duplicate detection algorithm for a dataset of ±1600 TVs, using locality-sensitive hashing (LSH) and gradient boosting (GB). Made for the individual assignment of the Computer Science for Business Analytics course (FEM21037) at Erasmus University Rotterdam, taught by Frasincar, F.

The file 'main.py' provides the general bootstrapping framework and must be run (with the other files in the same directory) to generate the results.

The file 'core.py' supplies the main steps of the duplicate detection algorithm, as outlined in the method section of the accompanying paper.

The file 'functions.py' defines several functions that are used in the several core steps of the algorithm, but are of a more technical nature.

Once the results have been generated, 'plots.py' will generate relevant plots for comparing the scores across hyperparameters.
