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
import re

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
        print('\nHere is a list of possible id_codes: \n', 
              possible_id_codes, '\n')
        id_user = input('Please input a relevant id_code from the csv: ')
        while id_user not in possible_id_codes:
            id_user = input("That is not a valid id_code, please try again: ")
        return param_user.loc[id_user]
    else:
        return param_user

def index_last(string, char):
    """provides the index of the last character rather than the first
    Inputs:
        string - any string
        char - a character - a string of length 1
    Returns:
        the final index of the char , -1 if the character is not in the string 
    """
    latest_char = -1
    for i in range(len(string)):
        if string[i] == char:
            latest_char = i
    return latest_char

index_last("Tim_is_cool", "_")
    
def deterministic_lists(analysis_type, param_user):
    """NEED TO WRITE IN IF AS THEY MAY NOT RUN IT #~
       Defines the different LTLI scenarios and which sets of parameters should
       be called for each of them
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
           being undertaken, must contain the key 'run_deterministic'
           param_user - df of parameters
       Returns:
           A dictionary with the scenario names and the set of parameters to
           be called for each
    """
    # Create scenario names based on parameter names
    param_names = param_user.iloc[0].index.values.tolist()
    scenario_names = []
    # Exclude disease proportions because they vary with disease burden
    # Exclude lives touched lives improved as they are results
    for i in param_names:
        if re.search('upper|lower', i):
            if not re.search('disease|touched|improved', i):
                scenario_names.append(i)
    # Add in the base case and placeholders for burden scenarioes
    scenario_names = ['base'] + scenario_names + ['burden_upper', 'burden_lower']
    
    # Create df indexes to select the right parameters for scenarios
    columns_to_select = []
    for i in param_names:
        if re.search('improved|touched', i):
            columns_to_select.append(i)
        elif not re.search('lower|upper|SD', i):
            columns_to_select.append(i)
    
    # Create a dictionary of scenario names and lists of relevant column names
    dict_of_columns = {}
    for i in scenario_names:
        # Don't change anything in the base case
        if i == 'base':
            new_columns = columns_to_select.copy()
        # Burden scenarios influence strain proportion, will use GBD ranges - not assumed ones
        elif re.search('burden', i):
            scenario_type = i[index_last(i, "_"):]
            new_columns = [re.sub('mean', scenario_type, i) 
                           if re.search('disease.*prop', i) else x 
                           for x in columns_to_select]
        # All others replace the column index for mean with upper or lower
        else:
            scenario_focus = i[:index_last(i, '_')] 
            column_name_to_replace = scenario_focus + '_mean'
            new_columns = [i if x == column_name_to_replace else x for x in columns_to_select]
        dict_of_columns[i] = new_columns.copy()
    return dict_of_columns

def scenario_param(param_user, scenarios_list, id_code):
    test_param = param_user.loc[id_code]    
    test_param = test_param.loc[scenarios_list]
    test_param_indexes = test_param.index.values.tolist()
    test_param_indexes = [re.sub('_mean', '', x) 
                          if re.search('mean', x) else x 
                          for x in test_param_indexes]
    test_param = test_param.reindex(test_param_indexes)
    return test_param 

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
    scenarios_dict = deterministic_lists(analysis_type, param_user)
    id_codes = param_user.index.values.tolist()
    id_code = id_codes[0]
    scenarios_list = scenarios_dict['base']
    scenarios_list2 = scenarios_dict['coverage_prob_upper']
    test_param1 = scenario_param(param_user, scenarios_list, id_code)  
    test_param2 = scenario_param(param_user, scenarios_list2, id_code)  
    
    
df = pd.DataFrame({'$a':[1,2], '$b': [10,20]})
df.columns = ['a', 'b']
df
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
