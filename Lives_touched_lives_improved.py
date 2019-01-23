# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 08:48:52 2019

Lives touched, lives improved model

@author: LaurencT
"""

# Import all functions required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set working directory and options
import os
os.chdir("C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/R/Lives touched lives improved/Data")

# Import parameters csv and import datasets

# A dictionary to determine the analysis type - update as prefered
analysis_type = {'run_all' : False,
                 'run_deterministic' : True,
                 'run_probabilistic' : True,
                 'num_trials' : 1000
                  } 

# Importing a csv of saved paramters and setting the id to be the index
param_user = pd.read_csv("LTLI_parameters_python.csv")
param_user = param_user.set_index("id_code") 

#

# Declaring all the functions used in the script
def check_run_all(analysis_type, param_user):
    """this is buggy as fuck, rewrite when you have your laptop #~
       Gets a user input of the right id_code if not running all previous 
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
           being undertaken, must contain the key 'run_all'
           param_user - df of parameters, must have 'id_code' as indexes
       Returns:
           the user code input by the user
     """
    if analysis_type['run_all'] == False:
        possible_id_codes = param_user.index.values.tolist()
        print_statement = print('\nHere is a list of possible id_codes: \n', possible_id_codes, '\n', 'Please input a relevant id_code from the csv: ')
        id_user = input(print_statement)
        while id_user not in possible_id_codes:
            id_user = input("That is not a valid id_code, please try again")
        return param_user.loc[id_user]
    else:
        return param_user

def restructure_to_deterministic(analysis_type, param_user):
    """Turns param_user matrix into a set of parameters for each of the deterministic
       and probabilistic analyses
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
           being undertaken, must contain the keys 'run_deterministic',
           'run_probabilistic' and 'num_trials'
           param_user - df of parameters, must have 'id_code' as indexes
       Returns:
           param - a df which only has the base case for each analysis
    """
# Code run sequentially

# Vary the parameter df depending on whether you are running all the analysis
# or just a subset
param_user = check_run_all(analysis_type, param_user)

# Create different versions of the analyses ready for sensitivity analyses
restructure_to_deterministic(analysis_type, param_user)

'2114750001A'





parameters

parameters["grant_number"]

parameters["grant_number"]==2

parameters["grant_number"][parameters["grant_number"]==2]

parameters["grant_number"].get_loc(2)

id_index = parameters["grant_number"][parameters["grant_number"]==2]

parameters["grant_number"].loc['3A']

id_index 

parameters.iloc[id_index]
