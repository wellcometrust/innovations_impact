This repository is for Innovations prospective impact modelling. 

The code is broken down into the following sections:

1)  Importing libraries and setting the working directory
2)  The analysis_type inputs which control scope of the analyses
3)  Importing data from csvs and basic cleaning of the data
4)  Functions to check the input parameters and datasets
5)  Functions which set up the deterministic scenarios to undertake deterministic sensitivity analysis (also known as One-at-a-time sensitivity analysis) https://en.wikipedia.org/wiki/Sensitivity_analysis#One-at-a-time_(OAT/OFAT)
    - this involves creating sets of parameters where each parameter is varied in turn to its upper or lower bound
6)  Functions which set up the probabilistic scenarios to undertake probabilistic sensitivity analysis (also known as Monte Carlo Simulation)
    - this involves creating sets of parameters where all the parameters is varied according to standard probability distributions simultaneously
7)  Based on these parameters, functions pick out the right data is drawn in for country level burden, coverage, population. The burden data are aggregated if there are multiple diseases or reductions to a subpopulation are made at this stage.
8)  Functions then adjust for the endemicity threshold, inflation_factor, intervention_cut (see methodology section)
9)  Functions now undertake some project specific adjustments. These are exceptions that would need to be generalised more fully over time. The model has functionality to limit the estimates by geography but it's not written in a generalisable form yet
10) Function calculate lives touched and lives improved for each set of parameters
11) Functions now split up the scenarios into deterministic and probabilistic scenarios
12) Functions then get the data into the right graph form and graph the results of the model
13) Functions then export the graphs into formatted slides, which are posted to a folder that should sync with Sharepoint
14) Functions then export the model estimates back to the original parameters csv, creating a time stamped backup
15) The if __main__=="__main__" is all of the functions then running in turn

Things to note:

Section 3 should be rewritten after the if __main__=="__main__" to make importing functions from this script quicker
Section 4 the checking functions have lists of variable names hard coded into them which need to be updated as more parameter options are added
Section 9 are exceptions that need to be generalised over time
Section 12 write graphs to a file path set at the top
Section 13 exports graphs to a file path that syncs with Sharepoint - you will need to create that sync to make it work https://wellcomecloud.sharepoint.com/sites/innovations/PAD/Forms/AllItems.aspx?id=%2Fsites%2Finnovations%2FPAD%2FLives%20touched%2C%20lives%20improved%20model%20results
Section 14 exports the model estimates to a file path

I have written a couple of tests for functions but did not write this with test driven development so they're not universal

Some of the functions still need doc strings, which I will add as a priority. 
