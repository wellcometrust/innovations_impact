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
                 'num_trials' : 1000
                  } 

# Importing a csv of saved paramters and setting the id to be the index
param_user_all = pd.read_csv("LTLI_parameters_python.csv")
param_user_all = param_user_all.set_index("id_code")
param_user_dict = {code : param_user_all.loc[code] 
                   for code in param_user_all.index.tolist()}

# Importing the relevant population data
population = pd.read_csv("GBD_population_2016_reshaped.csv")

# Subset the most relevant columns - this list could be updated to capture other 
# relevant age groups - they presently capture infants, young children, children/
# adolescents, working age people and then retired people
pop_columns = ['location_name', 'Both_<1 year', 'Both_1 to 4',
               'Both_5-14 years', 'Both_15-49 years', 'Both_50-69 years',
               'Both_70+ years']

population = population.loc[:, pop_columns]

# Merge two of the columns to make a more relevant one
population.insert(loc = 5, column =  'Both_15-69 years', 
                  value = (population['Both_15-49 years'] + population['Both_50-69 years']))

# Remove the merged columns 
pop_new_columns = [column for column in list(population) if not column in ['Both_15-49 years', 'Both_50-69 years']]

population = population[pop_new_columns]

# Rename columns to more consistent / intuitive names
pop_new_names = {'location_name' : 'country', 'Both_<1 year' : 'pop_0-0',
                 'Both_1 to 4' : 'pop_1-4', 'Both_5-14 years' : 'pop_5-14', 
                 'Both_15-69 years' : 'pop_15-69', 'Both_70+ years' : 'pop_70-100'}

population = population.rename(columns = pop_new_names)

# Reindex so country is the index
population = population.set_index(keys = 'country', drop = True)

#Importing the relevant disease burden data
burden_all = pd.read_csv('GBD_data_wide_2017.csv')

#Importing the relevant coverage data
coverage = pd.read_excel('C:/Users/laurenct/OneDrive - Wellcome Cloud/Health metrics/Vaccines data/Intervention penetration group placeholder.xlsm', 
                         sheet_name = 'Penetration assumptions')

coverage.columns = coverage.iloc[10]
coverage = coverage.iloc[11:, 1:]

cov_new_columns = [column for column in list(coverage) if re.search("cover|^country", column)]
coverage = coverage[cov_new_columns]

cov_new_names = {name : str.lower(re.sub(" ", "_", name)) for name in list(coverage)}
coverage = coverage.rename(columns = cov_new_names)

# Declaring all the functions used in the script
def check_run_all(analysis_type, param_user_dict):
    """Gets a user input of the right id_code if not running all previous 
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
           being undertaken, must contain the key 'run_all'
           param_user_all - df of parameters, must have 'id_code' as indexes
       Returns:
           the user code input by the user
    """
    id_codes = list(param_user_dict.keys())
    if analysis_type['run_all'] == False:
        print('\nHere is a list of possible id_codes: \n', id_codes, '\n')
        id_user = input('Please input a relevant id_code from the csv: ')
        while id_user not in id_codes:
            id_user = input("That is not a valid id_code, please try again: ")
        param_user_dict = {id_user: param_user_dict[id_user]} 
    return param_user_dict

def index_last(string, char):
    """provides the index of the last character rather than the first
    Inputs:
        string - any string
        char - a character - a string of length 1
    Returns:
        the index of the final appearance of char, 
        throws an error if char not in string to match behaviour of  str.index() 
    """
    latest_char = -1
    for i in range(len(string)):
        if string[i] == char:
            latest_char = i
    if latest_char == -1:
        raise ValueError(" substring not found")
    return latest_char

def lists_to_dict(list_keys, list_values):
    """two ordered lists to a dict where list_keys are the keys and list_values 
       are the values
       Inputs:
           list_keys - a list of n unique elements, where  n = 0, 1 or many
           list_values - a list of n elements
       Returns:
           dict_name - a dict of length n
    """
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
    """Defines the different LTLI scenarios and which sets of parameters should
       be called for each of them
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
           being undertaken, must contain the key 'run_deterministic'
           param_user - dict - keys: id_codes, values: series of parameters
       Returns:
           A dict - keys: scenario names, values - the set of parameters to
           be called for the corresponding scenario
    """
    # Create scenario names based on parameter names
    # This takes a random series from the dictionary's values to get the indexes
    # this is fine as all the values have the same indexes
    param_names = list(param_user.values())[0].index.tolist()
    scenario_names = [name for name in param_names if re.search('upper|lower', name)]
                      
    # Exclude disease proportions because they vary with disease burden
    scenario_names = [name for name in scenario_names if not re.search('disease.*prop', name)]
                      
    # Add in the base case and placeholders for burden scenarios
    scenario_names = ['base'] + scenario_names + ['burden_upper', 'burden_lower']
    
    # Create a list of parameter series indexes to select the right paramters 
    # for each of the scenarios
    columns_to_select = [column for column in param_names 
                         if not re.search('lower|upper|SD', column)]

    # Create a dictionary keys: scenario names, values: lists of relevant column names
    dict_of_columns = {}
    for scenario in scenario_names:
        # Don't change anything in the base case
        if scenario == 'base':
            new_columns = columns_to_select.copy()
        # Burden scenarios influence strain proportion, will use GBD ranges - 
        # not assumed ones so no adjustment factor for burden estimates yet
        elif re.search('burden', scenario):
            scenario_type = scenario[index_last(scenario, "_")+1:]
            new_columns = [re.sub('mean', scenario_type, column) 
                           if re.search('disease.*prop', column) else column 
                           for column in columns_to_select]
        # All others replace the column index for mean with upper or lower
        else:
            scenario_focus = scenario[:index_last(scenario, '_')] 
            column_name_to_replace = scenario_focus + '_mean'
            new_columns = [scenario 
                           if column == column_name_to_replace 
                           else column 
                           for column in columns_to_select]
        dict_of_columns[scenario] = new_columns.copy()
    return dict_of_columns

def scenario_param(param_user, param_list, id_code):
    """Subsets a series of parameters based on which are required for this 
       scenario, and renames the parameters to standard names (not upper, lower
       or mean anymore)
       Inputs:
           param_list: a list of parameter indexes required for this scenario
           param_user - dict - keys: id_codes, values: series of parameters
           id_code: a string used to index to get data on the right project
       Returns:
           param: a series that has been indexed and renamed
    """
    param = param_user[id_code]
    # series to get the relevant parameters for the scenario
    param = param.reindex(param_list)
    # get the names of those parameters and replace them with standard ones so 
    # all the parameters have standard names (no upper, lower, mean)
    param_indexes = param.index.values.tolist()
    param_indexes_new = [re.sub('_mean|_lower|_upper', '', x) 
                              if re.search('mean|lower|upper', x) else x 
                              for x in param_indexes]
    # generate mapping of old indexes to new indexes and use that to rename
    indexes_dict = lists_to_dict(param_indexes, param_indexes_new)
    param = param.rename(indexes_dict)
    # return the relevant parameters with standardised names
    return param 

def restructure_to_deterministic(analysis_type, param_user):
    """Turns param_user dict into a set of parameters for each of the 
       deterministic analyses
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
               being undertaken, must contain the keys 'run_deterministic' and
               'num_trials'
           param_user - dict - keys: id_codes, values: series of parameters
       Returns:
           param_dict - a dict- keys: id_codes, values dfs of parameters for 
               each of the deterministic scenarios
    """
    id_codes = list(param_user.keys())
    # generate a dictionary of which parameters are relevant for each scenario
    scenarios_dict = deterministic_lists(analysis_type, param_user)
    # create a dictionary where id_codes are the keys, and the values are dfs
    # of the parameter values for each scenario
    param_dict = {}
    for code in id_codes:
        # create a list to be filled with series, where each series is a set
        # of parameters for a scenario
        scenario_params = []
        for scenario in scenarios_dict.keys():
            param_list = scenarios_dict[scenario]
            param = scenario_param(param_user, param_list, code)
            param = param.rename(scenario)
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
        param_user - dict - keys: id_codes, values: series of parameters
    Returns:
        columns_names: a list of relevant column names for probabilistic 
            sensitivity analysis
    """
    # Gets all parameter names, indexes a dictionary which is fine as all values 
    # have the same indexes
    param_names = list(param_user.values())[0].index.tolist()
    column_names = [val for val in param_names if not re.search('upper|lower', val)]
    return column_names

def columns_to_add(column_names):
    """Subsets the list for elements that contain mean and then rename them
       so they aren't called mean, these will be the new names for columns 
       to be simluated probabilistically
       Inputs:
           column_names - a list of parameter names / df indexes 
       Returns:
           new_columns - a list of names of columns to be created
    """
    new_columns = [val for val in column_names if re.search('mean', val)]
    new_columns = [re.sub('_mean', '', val) for val in new_columns]
    return new_columns

def beta_moments(mean, sd):
    """Calculates the moments of a beta distribution based on mean and sd
       as they are more intuitive inputs
       More info on beta: https://en.wikipedia.org/wiki/Beta_distribution
       Inputs:
           mean - a float - the chosen mean of the distribution
           sd - a float - the chosen standard deviation of the distribution
       Returns:
           dict - keys: alpha and beta, values of corresponding moments
    """
    if (sd**2) >= (mean*(1 - mean)):
        raise ValueError('Variance (sd^2) must be less than (mean*(1-mean))')
    else:
        term = mean*(1 - mean)/(sd**2) - 1
        alpha = mean*term
        beta = (1 - mean)*term
        return {'alpha': alpha, 'beta': beta}
    
def gamma_moments(mean, sd):
    """Calculates the moments of a gamma distribution based on mean and sd
       as they are more intuitive inputs
       More info on gamma: https://en.wikipedia.org/wiki/Gamma_distribution
       Inputs:
           mean - a float - the chosen mean of the distribution
           sd - a float - the chosen standard deviation of the distribution
       Returns:
           dict - keys: shape and scale, values of corresponding moments
    """
    if mean < 0:
        raise ValueError('The mean must be above 0')
    else:
        scale = sd**2/mean
        shape = mean/scale
        return {'scale':scale, 'shape':shape}

def create_prob_df(param_prob, id_code, new_columns, num_trials):
    """Simulate new columns of paramters probabilistically based on assumed 
       means and sds, these columns are then added the data frames
       Inputs:
           param_prob - dict - keys: id_codes, values: the relevant parameters
               for probabilistic simulation
           id_code - the id_code for this project
           new_columns - list - names of all columns to be added
           num_trials - the number of trials to be simulated
       Returns:
           df - with the probabilistic and non-probabilistic columns
    """
    # Select the relevant parameters
    param_example = param_prob[id_code]
    # Create df and add repeated columns of non-prob param values
    prob_df = pd.DataFrame()
    for param in param_example.index.tolist():
        series = pd.Series(param_example.loc[param], index = range(1, num_trials + 1), name = param)
        prob_df = prob_df.append(series)
    # Transpose the df
    prob_df = prob_df.T
    # generate new columns based on probability distributions
    for column in new_columns:
        mean = float(param_example[column+'_mean'])
        sd = float(param_example[column+'_SD'])
        if sd == 0:
            data = np.array([mean for i in range(1,num_trials+1)])
        # use normal for things that vary around 1, inflation factor will need
        # changing probaby #~
        elif column in ['population', 'inflation_factor', 'coverage']:    
            data = norm.rvs(size = num_trials, loc = mean, scale = sd)
        # use beta distribution for all paramters that are a proportion
        elif column in ['disease_1_prop', 'disease_2_prop', 'disease_3_prop',
                        'coverage_prob', 'intervention_cut', 'share', 'efficacy']:
            data = beta.rvs(a = beta_moments(mean, sd)['alpha'], 
                            b = beta_moments(mean, sd)['beta'], 
                            size = num_trials)
        # use gamma for parameters that are non-negative and have a right skew
        elif column in ['endem_thresh']:
            data = gamma.rvs(a = gamma_moments(mean, sd)['shape'], 
                             scale = gamma_moments(mean, sd)['scale'], 
                             size = num_trials)
        # if a new parameter has been added will have to add it to one of the lists
        # above or this ValueError will be thrown every time
        else:
            raise ValueError(column, ' is an invalid column name')
        # turn the relevant new data into a series (which becomes a column)
        new_column = pd.Series(data, index = range(1, num_trials + 1), name = column)
        prob_df = pd.concat([prob_df, new_column.T], axis = 1, sort = False)
        # drop unnecessary columns
        columns_to_drop = [name for name in list(prob_df) if re.search("mean|SD", name)]
        prob_df = prob_df.drop(columns_to_drop, axis =1)
    return prob_df
    
def restructure_to_probabilistic(analysis_type, param_user):
    """Turns param_user dict into a set of parameters for each of the probabilistic
       sensitivity analyses
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
               being undertaken, must contain the keys 'run_probabilistic' and 
               'num_trials'
           param_user - dict - keys: id_codes, values: series of parameters
       Returns:
           param_dict - a dictionary with id_codes as keys and dfs of
               parameters for each of the probabilistic trials
    """
    # Remove parameters related to upper and lower bounds, focus on means and sds
    num_trials = analysis_type['num_trials']
    column_names = probabilistic_columns(param_user)
    new_columns = columns_to_add(column_names)
    param_prob = {k:v[column_names] for k, v in param_user.items()}
    # Generate the relevant id_codes
    id_codes = list(param_user.keys())
    param_dict = {code : create_prob_df(param_prob, code, new_columns, num_trials) 
                  for code in id_codes}
    return param_dict

def get_relevant_burden(deterministic_dict, burden_all):
    
    
# Code run sequentially

    
#Write in a parameter checking function as the first function#~
    
# Vary the parameter df depending on whether you are running all the analysis
# or just a subset
param_user = check_run_all(analysis_type, param_user_dict)

# Create different versions of the parameters ready for sensitivity analyses
deterministic_dict = restructure_to_deterministic(analysis_type, param_user)

# Create different versions of the parameters ready for sensitivity analyses
probabilistic_dict = restructure_to_probabilistic(analysis_type, param_user)


