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
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import beta
import re

# Set working directory and options
import os
os.chdir("C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/R/Lives touched lives improved/Data")

# Import parameters csv and import datasets

# A dictionary to determine the analysis type - update as prefered
analysis_type = {'run_all' : True,
                 'run_deterministic' : True,
                 'run_probabilistic' : True,
                 'num_trials' : 10
                  } 

# Importing a csv of saved paramters and setting the id to be the index
param_user_all = pd.read_csv("LTLI_parameters_python.csv")
param_user_all = param_user_all.set_index("id_code") 

#

# Declaring all the functions used in the script
def check_run_all(analysis_type, param_user):
    """Gets a user input of the right id_code if not running all previous 
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

def lists_to_dict(list_keys, list_values):
    """two ordered lists to a dict where list_keys are the keys and list_values 
       are the values
       Inputs:
           list_keys - a list of 0, 1 or many unique elements
           list_values - a list of the same number of elements
       Returns:
           dict_name - a dict the same length as the lists"""
    dict_name = {}
    if len(list_keys) > len(set(list_keys)):
        raise ValueError('The list of keys is not unique')
    elif len(list_keys) != len(list_values):
        raise ValueError('The lists are not the same length')
    else:
        for i in range(len(list_keys)):
            dict_name[list_keys[i]] = list_values[i]
        return dict_name

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
    try:
        param_names = param_user.iloc[0].index.values.tolist()
    except AttributeError:
        param_names = param_user.index.values.tolist()
    scenario_names = []
    # Exclude disease proportions because they vary with disease burden
    for i in param_names:
        if re.search('upper|lower', i):
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
        # Burden scenarios influence strain proportion, will use GBD ranges - 
        # not assumed ones so no adjustment factor for burden estimates yet
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
    """Subsets a series of parameters based on which are required for this 
       analysis, and renames the parameters to standard names (not upper, lower
       or mean anymore)
       Inputs:
           scenarios_list: a list of parameter indexes required for this scenario
           param_user: a series or dataframe of parameters to be subsetted
           id_code: a string used to index to get data on the right project
       Returns:
           test_param: a series that has been indexed and renamed
    """
    # try to look up the key if param is a dataframe if not it is already a series
    try:
        test_param = param_user.loc[id_code]
    except KeyError:
        test_param = param_user
    # series to get the relevant parameters for the scenario
    test_param = test_param.loc[scenarios_list]
    # get the names of those parameters and replace them with standard ones so 
    # all the parameters have standard names (no upper, lower, mean)
    test_param_indexes = test_param.index.values.tolist()
    test_param_indexes_new = [re.sub('_mean|_lower|_upper', '', x) 
                              if re.search('mean|lower|upper', x) else x 
                              for x in test_param_indexes]
    # generate mapping of old indexes to new indexes and use that to rename
    indexes_dict = lists_to_dict(test_param_indexes, test_param_indexes_new)
    test_param = test_param.rename(indexes_dict)
    # return the relevant parameters with standardised names
    return test_param 

def restructure_to_deterministic(analysis_type, param_user):
    """Turns param_user df into a set of parameters for each of the 
       deterministic analyses
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
           being undertaken, must contain the keys 'run_deterministic',
           'run_probabilistic' and 'num_trials'
           param_user - df of parameters, must have 'id_code' as indexes
       Returns:
           param_dict - a dictionary with id_codes as keys and dfs of
               parameters for each of the deterministic scenarioslysis
    """
    # generate a dictionary of which parameters are relevant for each scenario
    scenarios_dict = deterministic_lists(analysis_type, param_user)
    # generate a list of id_codes of projects to be analysed
    if analysis_type['run_all']:    
        id_codes = param_user.index.values.tolist()
    else:
        id_codes = [param_user.name]
    # create a dictionary where id_codes are the keys, and the values are dfs
    # of the parameter values for each scenario
    param_dict = {}    
    for code in id_codes:
        # create a list to be filled with series, where each series is a set
        # of parameters for a scenario
        scenario_params = []
        for scenario_name in scenarios_dict.keys():
            scenario_list = scenarios_dict[scenario_name]
            param = scenario_param(param_user, scenario_list, code)
            param = param.rename(scenario_name)
            scenario_params.append(param)
        # concatenate the series and transpose
        param_df = pd.concat(scenario_params, axis = 1, join_axes=[scenario_params[0].index])
        param_df = param_df.transpose()
        # populate the dictionary
        param_dict[code] = param_df
    return param_dict

def probabilistic_columns(param_user):
    """Based on user inputted parameters, takes the parameters and subsets the 
    relevant parameters for the probabilistic sensitivity analysis
    Inputs:
        param_user: a df or series of user inputted parameters for either one
            or multiple projects
    Returns:
        columns_names: a list of relevant column names for probabilistic sensiti
            vity analysis
    """
    # Find relevant column names for probabilistic sensitivity analysis
    try:
        param_names = param_user.iloc[0].index.values.tolist()
    except AttributeError:
        param_names = param_user.index.values.tolist()
    column_names = [val for val in param_names if not re.search('upper|lower', val)]
    return column_names

def columns_to_add(column_names):
    new_columns = [val for val in column_names if re.search('mean', val)]
    new_columns = [re.sub('_mean', '', val) for val in new_columns]
    return new_columns

def beta_moments(mean, sd):
    if (sd**2) >= (mean*(1 - mean)):
        raise ValueError('Variance (sd^2) must be less than (mean*(1-mean))')
    else:
        term = mean*(1 - mean)/(sd**2) - 1
        alpha = mean*term
        beta = (1 - mean)*term
        return {'alpha': alpha, 'beta': beta}
    
def gamma_moments(mean, sd):
    if mean < 0:
        raise ValueError('The mean must be above 0')
    else:
        scale = sd**2/mean
        shape = mean/scale
        return {'scale':scale, 'shape':shape}


def trial_param(param_prob, id_code, column_names):
    # try to look up the key if param is a dataframe if not it is already a series
    try:
        test_param = param_prob.loc[id_code]
    except KeyError:
        test_param = param_prob
    new_columns = columns_to_add(column_names)
    column = new_columns[0]
    for column in new_columns:
        mean = float(test_param[column+'_mean'])
        sd = float(test_param[column+'_SD'])
        if sd == 0:
            test_param.at[column] = mean
        elif column in ['population', 'inflation_factor', 'coverage']:    
            test_param.at[column] = float(norm.rvs(size = 1, loc = mean, scale = sd))
        elif column in ['disease_1_prop', 'disease_2_prop', 'disease_3_prop',
                        'coverage_prob', 'intervention_cut', 'share', 'efficacy']:
            test_param.at[column] = float(beta.rvs(a = beta_moments(mean, sd)['alpha'], 
                                                   b = beta_moments(mean, sd)['beta'], size = 1))
        elif column in ['endem_thresh']:
            test_param.at[column] = float(gamma.rvs(a = gamma_moments(mean, sd)['shape'], 
                                                    scale = gamma_moments(mean, sd)['scale'], size = 1))
        else:
            raise ValueError(column, ' is an invalid column name')
    return test_param
    
def restructure_to_probabilistic(analysis_type, param_user):
    """Turns param_user df into a set of parameters for each of the probabilistic
       sensitivity analyses
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
               being undertaken, must contain the keys 'run_probabilistic' and 
               'num_trials'
           param_user - df of parameters, must have 'id_code' as indexes
       Returns:
           param_dict - a dictionary with id_codes as keys and dfs of
               parameters for each of the probabilistic trials
    """
    # Remove parameters related to upper and lower bounds, focus on means and sds
    column_names = probabilistic_columns(param_user)
    try:
        param_prob = param_user.loc[column_names]
    except KeyError:
        param_prob = param_user[column_names]
    if analysis_type['run_all']:    
        id_codes = param_user.index.values.tolist()
    else:
        id_codes = [param_user.name]
    param_dict = {}
    for code in id_codes:
        all_trial_params = []
        for trial in range(analysis_type['num_trials']):
            trial_params = trial_param(param_prob, code, column_names)
            trial_params = trial_params.rename(trial)
            all_trial_params.append(trial_params)
        param_df = pd.concat(all_trial_params, axis = 1, join_axes=[all_trial_params[0].index])
        param_df = param_df.transpose()
        param_dict[code] = param_df
    return param_dict
    
# Code run sequentially

# Vary the parameter df depending on whether you are running all the analysis
# or just a subset
param_user = check_run_all(analysis_type, param_user_all)

# Create different versions of the analyses ready for sensitivity analyses
deterministic_dict = restructure_to_deterministic(analysis_type, param_user)

# Create different versions of the analyses ready for sensitivity analyses
probabilistic_dict = restructure_to_probabilistic(analysis_type, param_user)


