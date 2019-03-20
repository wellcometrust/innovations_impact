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
import datetime

# Set working directory and options
import os
os.chdir('C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/data')
directory = 'C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/graphs/'


# Import parameters csv and import datasets

# A dictionary to determine the analysis type - update as prefered
analysis_type = {'run_all' : False,
                 'run_deterministic' : True,
                 'run_probabilistic' : True,
                 'num_trials' : 1000,
                 'overwrite_parameters' : False
                  } 

# Importing a csv of saved paramters and setting the id to be the index
param_user_all = pd.read_csv('LTLI_parameters_python.csv')
param_user_all = param_user_all.set_index('id_code')
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
pop_new_columns = [column for column in list(population) 
                   if not column in ['Both_15-49 years', 'Both_50-69 years']]

population = population[pop_new_columns]

# Rename columns to more consistent / intuitive names
pop_new_names = {'location_name' : 'country', 'Both_<1 year' : 'pop_0-0',
                 'Both_1 to 4' : 'pop_1-4', 'Both_5-14 years' : 'pop_5-14', 
                 'Both_15-69 years' : 'pop_15-69', 'Both_70+ years' : 'pop_70-100'}

population = population.rename(columns = pop_new_names)

# Reindex so country is the index
population = population.set_index(keys = 'country', drop = True)

# Importing the relevant disease burden data
burden_all = pd.read_csv('GBD_data_wide_2017.csv')

burden_all.columns = [column.lower() for column in burden_all.columns.tolist()]

# Importing the relevant coverage data
coverage = pd.read_excel('C:/Users/laurenct/OneDrive - Wellcome Cloud/Health metrics/Vaccines data/Intervention penetration group placeholder.xlsm', 
                         sheet_name = 'Penetration assumptions')

coverage.columns = coverage.iloc[10]
coverage = coverage.iloc[11:, 1:]

cov_new_columns = [column for column in list(coverage) if re.search('cover|^country', column)]
coverage = coverage[cov_new_columns]

cov_new_names = {name : str.lower(re.sub(" ", "_", name)) for name in list(coverage)}
coverage = coverage.rename(columns = cov_new_names)
coverage.index = coverage['country']

# Declaring all the functions used in the script
def check_analysis_type(analysis_type):
    """Checks the values in the analysis type dictionary to make sure they are
       bools or ints as appropriate
       Inputs:
           analysis_type: a dict where values are ints or bools depending on
               what they determine
    """
    # Take out the values that are supposed to be bools
    analysis_values = [analysis_type['run_all'], 
                       analysis_type['run_deterministic'],
                       analysis_type['run_probabilistic'],
                       analysis_type['overwrite_parameters']]
    # Test to see if they are bools
    for value in analysis_values:
        if not type(value) is bool:
            raise ValueError('all of the run_... parameters in analysis type should be booleans')
    # Test to make sure num trials is an int
    if not type(analysis_type['num_trials']) is int:
        raise ValueError('the number of trials has to be an integer')

def check_indexes(param_user_all):
    """Checks to ensure all of the id_codes for each separate analysis are unique
       Inputs:
           param_user_all - a df of user inputted parameters
    """
    indexes = param_user_all.index.tolist()
    if len(set(indexes)) < len(indexes):
        raise ValueError('id_codes should be unique, please ensure there are no duplicate id codes in the parameters csv')

def check_columns(param_user_all):
    """Checks to ensure all of the required column names are contained in the 
       parameters tab
       Inputs:
           param_user_all - a df of user inputted parameters

    """
    necessary_columns = ['id_name', 'intervention_type', 'disease_1', 'disease_2', 
                         'disease_3', 'age', 'population_upper', 
                         'population_mean', 'population_lower', 'population_SD', 
                         'disease_1_prop_mean', 'disease_1_prop_lower', 
                         'disease_1_prop_upper', 'disease_1_prop_SD', 
                         'disease_2_prop_mean', 'disease_2_prop_lower', 
                         'disease_2_prop_upper', 'disease_2_prop_SD', 
                         'disease_3_prop_mean', 'disease_3_prop_lower', 
                         'disease_3_prop_upper', 'disease_3_prop_SD', 
                         'endem_thresh_mean', 'endem_thresh_lower', 
                         'endem_thresh_upper', 'endem_thresh_SD', 
                         'endem_thresh_metric', 'inflation_factor_upper', 
                         'inflation_factor_mean', 'inflation_factor_lower', 
                         'inflation_factor_SD', 'intervention_cut_upper', 
                         'intervention_cut_mean', 'intervention_cut_lower', 
                         'intervention_cut_SD', 'prob_cover_upper', 
                         'prob_cover_mean', 'prob_cover_lower', 
                         'prob_cover_SD', 'coverage_upper', 'coverage_mean', 
                         'coverage_lower', 'coverage_SD', 'coverage_below_threshold', 
                         'share_upper', 'share_mean', 'share_lower', 'share_SD', 
                         'efficacy_upper', 'efficacy_mean', 'efficacy_lower', 
                         'efficacy_SD', 'lives_touched', 'lives_touched_975', 
                         'lives_touched_025', 'lives_improved', 'lives_improved_975', 
                         'lives_improved_025', 'exception_count', 'exception_comment']
    necessary_columns_missing = [column for column in necessary_columns
                                 if column not in list(param_user_all) ]
    if len(necessary_columns_missing)>0:
        raise ValueError('The following columns are missing from the parameter csv '+str(necessary_columns_missing))

def check_iterable_1_not_smaller(iterable_1, iterable_2):
    """Checks two iterables of the same length for whether each element in 1
       is at least as big as the corresponding element of 2
       Inputs:
           iterable_1 - an iterable of arbitary length n
           iterable_2 - an iterable of length n which is ordered to correspond 
               to iterable_1
       Returns:
           bool reflecting whether all elements are not smaller
    """
    if len(iterable_1) == len(iterable_2):
        bool_list = map(lambda x,y: x>=y, iterable_1, iterable_2)        
    else:
        raise ValueError("the iterables must be the same length")
    return all(list(bool_list))

def check_upper_lower(param_user_all):
    """Checks to ensure every value of the upper parameters is at least as great
       as the mean and the mean is at least as great as the lower
       inputs:
           param_user_all - a df of user inputted parameters
    """
    column_roots = [re.sub('_mean', '', column)
                    for column in param_user_all if re.search('mean', column)]
    column_mapping = [(column+"_upper", column+"_mean", column+"_lower") 
                       for column in column_roots]
    for mapping in column_mapping:
        # Check upper >= mean
        if not check_iterable_1_not_smaller(iterable_1 = param_user_all[mapping[0]],
                                            iterable_2 = param_user_all[mapping[1]]):
            raise ValueError(mapping[1]+' is greater than '+mapping[0])            
        # Check mean >= lower
        elif not check_iterable_1_not_smaller(iterable_1 = param_user_all[mapping[1]],
                                            iterable_2 = param_user_all[mapping[2]]):
            raise ValueError(mapping[2]+' is greater than '+mapping[1])
        # Upper>=lower by transitivity therefore is valid
        else:
            print('Values of '+mapping[0]+', '+mapping[1]+', '+mapping[2]+' are consistent')

def column_checker(column, ther_diag_param, correct_value):
    """Checks if a column of a df is all one value, raises an error if it's not
       inputs:
           column - a string which is the name of a column in ther_diag_param
           ther_diag_param - a df of parameters that is filtered so it only
               contains entries on diagnostics and therapeutics
           correct_value - a number that is the expected column value
    """
    if not all(v == correct_value for v in ther_diag_param[column]):
        raise ValueError('Values in '+column+' should all be '+str(correct_value)+' for diagnostics and therapeutics')
    
def check_diag_ther(param_user_all):
    """Checks diagnostics and therapeutics to ensure they don't have incorrect 
       values for population and endemicity - as those do not feed into these 
       analyses
       Inputs:
           param_user_all - a df of all user inputted parameters
    """
    # Filter the df so it is relevant to therapeutics and diagnostics
    ther_diag_param = param_user_all[param_user_all.intervention_type.isin(['Therapeutic', 'Therapeutic mental health', 'Diagnostic'])]
    # Find the parameters that aren't relevant for the analysis and make sure they won't affect results
    population_columns = [column for column in param_user_all if re.search('population', column)]
    endem_columns = [column for column in param_user_all if re.search('endem.*[d-z]$', column)]
    # Check each column in turn to ensure it is the right value
    for column in population_columns:
        # Population estimates don't feed in to these analysis so shouldn't be 
        # have a distribution
        if re.search("SD", column):                    
            column_checker(column, ther_diag_param, correct_value = 0)
        # Population estimates don't feed in to these analysis so should all be 1 
        else:
            column_checker(column, ther_diag_param, correct_value = 1)
        # Endemicity threshold shouldn't be applied here, as countries with 
        # low burden may use it, on their limited number of patients
    for column in endem_columns:
        column_checker(column, ther_diag_param, correct_value = 0)

def check_disease_selection(param_user_all, burden_all):
    """Checks to confirm that all selected diseases are valid
       Inputs:
           param_user_all - a df of inputted parameters must contain disease_1
               disease_2, and disease_3 columns
           burden_all - a df of burden data, disease column must be called cause
    """
    # All diseases in the dataset - GBD+aetiology+malaria breakdown#~ - Empty is 
    # only other valid option
    valid_diseases = set(burden_all['cause'].tolist()+['Empty'])
    # Makes a list of all diseases inputted by the user
    disease_choices = (param_user_all['disease_1'].tolist()+
                       param_user_all['disease_2'].tolist()+
                       param_user_all['disease_3'].tolist())
    for disease in disease_choices:
        if not disease in valid_diseases:
            #~ could write in a fuzzy lookup for potential valid names
            raise ValueError(disease+' is not a valid disease name, please search XX for valid diseases')

def check_burden(burden_all):
    """Checks to make sure all the column names in the burden data are consist
       ent with what are called in the later code
       Inputs:
           burden_all - names of disease burden columns contain a _ because 
               they are 'measure_metric', there are columns for 'country', 
               'age', 'cause'
    """
    expected_columns = ['age', 'country', 'cause']
    column_name_check = [column in list(burden_all) for column in expected_columns]
    if not all(column_name_check):
        raise ValueError('check that age, country and cause are in burden all')
    pass #~write this in when analysis is written - see what columns are required

def check_population(population):
    """Checks to make sure all the column names and indexes in the population 
       data are consistent with what are called in the later code
       Inputs:
           population - a df of population data
    """
    # Check the column names are there
    expected_names = ['pop_0-0', 'pop_1-4', 'pop_5-14', 'pop_15-69','pop_70-100']
    expected_names_present = [name in list(population) for name in expected_names]
    if not all(expected_names_present):
        raise ValueError('The population df does not have the expected columns')
    # Check the index for country names
    elif 'France' not in population.index:
        raise ValueError('The population df does not have countries as indexes')
    else:
        # do nothing because not problems
        pass 

def check_coverage(coverage):
    """Checks to make sure all the column names and indexes in the coverage data 
       are consistent with what are called in the later code
       Inputs:
           population - a df of coverage data
    """
    expected_names = ['vaccine_coverage',  'vaccine_prob_cover',
                     'diagnostic_coverage', 'diagnostic_prob_cover',
                     'therapeutic_coverage', 'therapeutic_prob_cover',
                     'therapeutic_mental_health_coverage', 
                     'therapeutic_mental_health_prob_cover']
    expected_names_missing = [name for name in expected_names
                              if name not in list(coverage)]
    if len(expected_names_missing)>0:
        raise ValueError('The following columns are missing from the population df '+str(expected_names_missing))
    # Check the index for country names
    elif 'France' not in coverage.index:
        raise ValueError('The population df does not have countries as indexes')
    else:
        # do nothing because not problems
        pass 

def check_inputs(analysis_type, param_user_all, population, coverage, burden_all):
    """Checks all user inputs using a variety of functions - raises error if 
       not valid inputs
       Inputs:
           analysis_type - a dict 
           param_user_all - a df of input parameters
           population - a df of population data
           coverage - a df of coverage data
           burden_all - a df of burden data
    """
    # Make sure user inputted parameters are valid
    check_analysis_type(analysis_type)
    check_indexes(param_user_all)
    check_columns(param_user_all)
    check_upper_lower(param_user_all)
    check_diag_ther(param_user_all)
    check_disease_selection(param_user_all, burden_all)
    # Make sure burden / population / coverage datasets imported have the 
    # right country names and column names
    check_burden(burden_all)
    check_population(population)
    check_coverage(coverage)
    
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
           a dict of length n
    """
    if len(list_keys) > len(set(list_keys)):
        raise ValueError('The list of keys is not unique')
    elif len(list_keys) != len(list_values):
        raise ValueError('The lists are not the same length')
    else:
        return {k:v for k,v in zip(list_keys, list_values)}

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
    # Series to get the relevant parameters for the scenario
    param = param.reindex(param_list)
    # Get the names of those parameters and replace them with standard ones so 
    # all the parameters have standard names (no upper, lower, mean)
    param_indexes = param.index.values.tolist()
    param_indexes_new = [re.sub('_mean|_lower|_upper', '', x) 
                         if re.search('mean|lower|upper', x) else x 
                         for x in param_indexes]
    # Generate mapping of old indexes to new indexes and use that to rename
    indexes_dict = lists_to_dict(param_indexes, param_indexes_new)
    param = param.rename(indexes_dict)
    # Return the relevant parameters with standardised names
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
    # Generate a dictionary of which parameters are relevant for each scenario
    scenarios_dict = deterministic_lists(analysis_type, param_user)
    # Create a dictionary where id_codes are the keys, and the values are dfs
    # of the parameter values for each scenario
    param_dict = {}
    for code in id_codes:
        # Create a list to be filled with series, where each series is a set
        # of parameters for a scenario
        scenario_params = []
        for scenario in scenarios_dict.keys():
            param_list = scenarios_dict[scenario]
            param = scenario_param(param_user, param_list, code)
            param = param.rename(scenario)
            scenario_params.append(param)
        # Concatenate the series and transpose
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
    # Generate new columns based on probability distributions
    for column in new_columns:
        mean = float(param_example[column+'_mean'])
        sd = float(param_example[column+'_SD'])
        if sd == 0:
            data = np.array([mean for i in range(1,num_trials+1)])
        # Use normal for things that vary around 1, inflation factor will need
        # changing probaby #~
        elif column in ['population', 'inflation_factor', 'share',
                        'coverage', 'prob_cover']:    
            data = norm.rvs(size = num_trials, loc = mean, scale = sd)
        # Use beta distribution for all paramters that are a proportion
        elif column in ['disease_1_prop', 'disease_2_prop', 'disease_3_prop',
                        'intervention_cut', 'efficacy']:
            data = beta.rvs(a = beta_moments(mean, sd)['alpha'], 
                            b = beta_moments(mean, sd)['beta'], 
                            size = num_trials)
        # Use gamma for parameters that are non-negative and have a right skew
        elif column in ['endem_thresh']:
            data = gamma.rvs(a = gamma_moments(mean, sd)['shape'], 
                             scale = gamma_moments(mean, sd)['scale'], 
                             size = num_trials)
        # If a new parameter has been added will have to add it to one of the lists
        # above or this ValueError will be thrown every time
        else:
            raise ValueError(column, ' is an invalid column name')
        # Turn the relevant new data into a series (which becomes a column)
        new_column = pd.Series(data, index = range(1, num_trials + 1), name = column)
        prob_df = pd.concat([prob_df, new_column.T], axis = 1, sort = False)
        # Drop unnecessary columns
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
    param_dict = {code: create_prob_df(param_prob, code, new_columns, num_trials) 
                  for code in id_codes}
    return param_dict

def get_relevant_burden(param_dict, burden_all):
    """Return a dict of burden data frames with the relevant conditions isolated
       Inputs:
           param_dict - keys: id_codes, values dfs of parameters for 
               each of the trials / scenarios
           burden_all - a df of burden data, must have the columns cause and age
       Returns:
           a dict - keys: id_codes, values dfs of burden data
    """
    # Create dictionary of lists to filter disease column by
    disease_lists = {k: [param_dict[k]['disease_1'][0],
                         param_dict[k]['disease_2'][0],
                         param_dict[k]['disease_3'][0]]
                     for k in param_dict.keys()}
    # Create dictionary of burden dfs
    burden_dict = {k: burden_all[burden_all['cause'].isin(disease_lists[k])]
                   for k in disease_lists.keys()}
    # Filter based on age
    burden_dict = {k: burden_dict[k][burden_dict[k]['age'] == param_dict[k]['age'][0]]
                   for k in burden_dict.keys()}
    return burden_dict

def select_columns_burden(burden_df, index):
    """This probabilistically varies columns by GBD ranges for probabilistic 
       trials it selects the correct deterministic columns for deterministic 
       trials and subsets and renames the columns.
       Inputs:
           burden_df - a df of burden data with upper, lower and mean estimates
               for at least one burden measure
           index - a string to indicate which deterministic scenario or which
               trial it is
       Returns:
           a df of burden data with just one column for each measure
    """
    new_burden_df = burden_df.copy()
    # Create column roots e.g. DALY_rate
    column_roots = [re.sub('_mean', '', column) 
                    for column in list(new_burden_df) 
                    if re.search('mean', column)]
    # Vary relevant column deterministically or probabilistically based on its root
    for root in column_roots:    
        try:
            int(index)
            #~ changed from normal to gamma
            mean = new_burden_df[root + '_mean']
            sd = (new_burden_df[root + '_mean']-new_burden_df[root + '_lower'])/2
            new_burden_df[root + '_mean'] = gamma.rvs(a = gamma_moments(mean, sd)['shape'], 
                             scale = gamma_moments(mean, sd)['scale'], 
                             size = 1)
            new_burden_df[root + '_mean'] = np.where(new_burden_df[root + '_mean'] <0, 
                                            0, new_burden_df[root + '_mean'])
        except ValueError:
            if index == 'burden_lower':
                new_burden_df[root + '_mean'] = new_burden_df[root + '_lower']
            elif index == 'burden_upper':
                new_burden_df[root + '_mean'] = new_burden_df[root + '_upper']
        
    # Remove upper and lower columns as the relevant column is now the mean
    relevant_columns = [column for column in list(new_burden_df) if not re.search('upper|lower', column)]
    new_burden_df = new_burden_df[relevant_columns]
    # Create new column name mapping
    new_column_names_dict = {column: re.sub('_mean', '', column) for column in relevant_columns}  
    # Rename columns
    new_burden_df = new_burden_df.rename(columns = new_column_names_dict)
    return new_burden_df
    
def strain_adjust_burden(burden_df, param_df, index):
    """Adjusts a data frame of disease burden downwards in case not all of the
       patients would benefit from the interventions (due to sub disease strains
       or other factors)
       Inputs:
           burden_df - a df with columns of disease burden where the names of 
               all of those columns contain a _ because they are 'measure_metric', 
               there is also a cause column 
           param_dict - a df of parameters with columns for 3 diseases in the form
               disease_[1-3] and with corresponding strain proportions for each of
               those diseases in the form disease_[1-3]_prop
           index - a string to indicate which deterministic scenario or which
               trial it is
       Returns:
           a burden df that has been adjusted for those substrains     
    """
    # Copy the burden_df to avoid side effects
    new_burden_df = burden_df.copy()
    # Create a list of burden columns because they are the form 'measure_metric'
    column_list = [column for column in list(new_burden_df) if re.search('_', column)]
    # Create a mapping of the disease and proportion
    disease_dict = {k: k+"_prop" for k in ['disease_1', 'disease_2', 'disease_3']}
    # Adjust the df for the substrain prop for each disease
    for column in column_list:
        for disease in disease_dict.keys():
            new_burden_df[column] = np.where(new_burden_df['cause'] == param_df.loc[index, disease], 
                                         new_burden_df[column]*param_df.loc[index, disease_dict[disease]], 
                                         new_burden_df[column])
    return new_burden_df

def aggregate_burden(burden_df):
    """Sums the burden from various conditions targetted by the intervention
       Inputs:
           burden_df - a df with columns for country, age, cause and disease
               burden
       Returns:
           a df where the cause is now renamed to be a list of causes and the
           burden from original causes is now aggregated
    """
    # Create a copy of burden_df to avoid side effects
    new_burden_df = burden_df.copy()    
    # Create list then string of causes
    causes = set(new_burden_df['cause'])
    new_cause_name = ', '.join(causes)
    # Aggregate by age and couttry
    new_burden_df= new_burden_df.groupby(['country', 'age']).sum()
    # Turn the indexes back into columns in a df
    index_list = new_burden_df.index.tolist()
    summed_df = pd.DataFrame(index_list, columns = ['country', 'age'], index = index_list)
    # Add the new cause name to the df
    summed_df['cause'] = new_cause_name
    # Merge the burden with the country, age group, and cause columns
    summed_df = pd.concat([summed_df, new_burden_df], axis = 1)
    # Reindex so the country is the only index
    summed_df.index = [i[0] for i in summed_df.index.tolist()]
    return summed_df
    
def adjust_burden(burden_dict, param_dict):
    """Adjusts the burden numbers down for substrains
       Inputs:
           burden_dict - a dict - keys are id_codes for the projects
               values are dataframes of all relevant burden for those projects
           param_dict - a dict - keys are id_codes for the projects and values 
               are dfs of parameters for the different scenarios and trials
       Returns:
           a dictionary keys are the id_codes for projects, values are dicts
           each trial key responding to a homogenously proportioned burden df
           with the relevant figure for that project / trial
    """
    burden_dict_new = {}
    # Loop through each of the id_codes 
    for code in param_dict.keys():
        # For each code find the relevant burden data and parameters 
        burden_scenarios_dict = {}
        burden_df = burden_dict[code].copy()
        param_df = param_dict[code].copy()
        # For each trial adjust the burden data to make sure it is relevant
        # for the trial
        for index in param_df.index.tolist():
            burden_df_index = select_columns_burden(burden_df, index)
            burden_df_index = strain_adjust_burden(burden_df_index, param_df, index)
            burden_df_index = aggregate_burden(burden_df_index)
            burden_scenarios_dict[index] = burden_df_index            
        # Add the new dictionary for the trials to the outer dictionary
        burden_dict_new[code] = burden_scenarios_dict
    return burden_dict_new

def create_coverage_population_dict(coverage, population, param_dict):
    """Creates a dictionary where keys are id_codes and values are dfs
       of relevant coverage / population data for the intervention
       Inputs:
           coverage - a df of coverage data containing the columns 'country' and
               coverage and prob_cover columns for each modality
           population - a df if population data with the columns 'country' and 
               population columns
           param_dict - a df with the column 'intervention_type'
       Returns:
           dict where keys are id_codes and values are dfs of relevant coverage 
               / population data for the intervention        
    """
    cov_pop_dict = {}
    # Loop through the id_codes to be keys in the dictionary
    for code in param_dict.keys():
        new_coverage = coverage.copy()
        population = population.copy()   
        # Select therapeutic coverage columns if it is a Therapeutic (therapeutic)
        if param_dict[code]['intervention_type'][0] == 'Therapeutic':
            new_coverage = new_coverage[['country',
                                 'therapeutic_coverage', 
                                 'therapeutic_prob_cover']]
        # Select therapeutic mental health coverage columns if it is a therapeutic
        # for a mental health condition
        elif param_dict[code]['intervention_type'][0] == 'Therapeutic mental health':
            new_coverage = new_coverage[['country', 
                                 'therapeutic_mental_health_coverage', 
                                 'therapeutic_mental_health_prob_cover']]
        # Select diagnostics coverage columns if it is a diagnostic
        elif param_dict[code]['intervention_type'][0] == 'Diagnostic':
            new_coverage = new_coverage[['country', 
                                 'diagnostic_coverage', 
                                 'diagnostic_prob_cover']]
        # Select vaccine coverage columns if it is a vaccine
        elif param_dict[code]['intervention_type'][0] == 'Vaccine':
            new_coverage = new_coverage[['country', 
                                 'vaccine_coverage', 
                                 'vaccine_prob_cover']]
        else:
            raise ValueError('The value of intervention_type for '+code+' is not valid')
        # Create new column names and rename
        new_column_names = {column: re.sub('vaccine_|diagnostic_|therapeutic_mental_health_|therapeutic_', '', column) 
                            for column in list(new_coverage)}
        new_coverage = new_coverage.rename(columns = new_column_names)
        # Merge the coverage and population data
        cov_pop_df = pd.concat([new_coverage, population], axis = 1)
        cov_pop_dict[code] = cov_pop_df
    return cov_pop_dict

def adjust_cov_pop_df(cov_pop_df, index, param_df):
    """Adjust the coverage and proportion based on the parameters for each scenario
       Inputs:
           cov_pop_df - a df with population columns and coverage columns in the
               form prob_cover
           index - a string to indicate which deterministic scenario or which
               trial it is
           param_df - a df with parameters for each of the scenarios must contain
               columns: 'coverage' and 'population'
       Returns:
           a df with the population and coverage columns adjusted
    """
    new_cov_pop_df = cov_pop_df.copy()
    # Adjust population columns by the population assumption for this scenario
    pop_columns = [column for column in list(new_cov_pop_df) if re.search('pop', column)]
    for column in pop_columns:
        new_cov_pop_df[column] = new_cov_pop_df[column]*param_df.loc[index, 'population']
    # Adjust coverage columns by the coverage assumption for this scenario
    cov_columns = [column for column in list(new_cov_pop_df) if re.search('coverage', column)]
    for column in cov_columns:
        new_cov_pop_df[column] = new_cov_pop_df[column]*param_df.loc[index, 'coverage']
        new_cov_pop_df[column] = np.where(new_cov_pop_df[column] > 0.95, 
                                          0.95, new_cov_pop_df[column])
    # Adjust prob_cover columns by the coverage assumption for this scenario
    cov_columns = [column for column in list(new_cov_pop_df) if re.search('prob_cover', column)]
    for column in cov_columns:
        new_cov_pop_df[column] = new_cov_pop_df[column]*param_df.loc[index, 'prob_cover']
        new_cov_pop_df[column] = np.where(new_cov_pop_df[column] > 0.95, 
                                          0.95, new_cov_pop_df[column]) 
    return new_cov_pop_df  

def adjust_cov_pop_for_trials(cov_pop_dict, param_dict):
    """Adjusts the cov_pop_dict so its values are now a dictionary of scenario
       and dfs to use in each of those scenarios
       Inputs:
           cov_pop_dict - a dictionary where the keys are id_codes and the 
               values are dfs of coverage and population data
           param_dict - a dictionary where the keys are id_codes and the values
               are dfs of paramters for each of hte scenarios
       Returns:
           a dict where the keys are id_codes and the values are dicts of where
           keys are scenarios and values are dfs of appropriate coverage / population
           data
    """
    new_cov_pop_dict = {}
    # Loop through the dictionary by keys
    for code in cov_pop_dict.keys():
        # Set up the dict and dfs
        cov_pop_scenarios_dict = {}
        cov_pop_df = cov_pop_dict[code]
        param_df = param_dict[code]
        # Loop through each of the scenarios 
        for index in param_df.index.tolist():
            # Adjust the cov_pop_df based on the scenario parameters
            new_cov_pop_df = adjust_cov_pop_df(cov_pop_df, index, param_df)
            cov_pop_scenarios_dict[index] = new_cov_pop_df.copy()
        new_cov_pop_dict[code] = cov_pop_scenarios_dict    
    return new_cov_pop_dict
    
def merge_cov_pop_and_burden(burden_dict, cov_pop_dict):
    """Merges the dataframes within a nested dictionary structure
       
       Inputs:
           burden_dict - keys - id_code, values are dictionary of scenarios
               and burden data dfs
           cov_pop_dict - keys - id_code, values are dictionary of scenarios
               and coverage and population data dfs
           (both sets of keys, and indexes of the dfs have to be equivalent)
       Returns:
           a merged data_dict of dictionaries
    """
    data_dict = {}
    for code in burden_dict.keys():
        scenario_dict = {}
        scenario_dict_burden = burden_dict[code]
        scenario_dict_cov_pop = cov_pop_dict[code]
        for scen in scenario_dict_burden.keys():
             # Merge population and burden data
             merged_df = pd.concat([scenario_dict_burden[scen],
                                            scenario_dict_cov_pop[scen]],
                                            axis = 1)
             # Deduplicate columns from the df
             merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
             scenario_dict[scen] = merged_df
        data_dict[code] = scenario_dict
    return data_dict

def create_target_population(cov_pop_burden_df, param_df, index):
    """Adds a new column to cov_pop_burden_df which is the target population
       Inputs:
           cov_pop_burden_df - a df which must contain 'incidence_number' and 
               'pop_0-0' columns
           param_df - a df of parameters which must contain the column 'intervention_type'
               where all of the values are 'Therapeutic', 'Diagnostic' or 'Vaccine'
           index - a string that is one of the indexes of param_df
       Returns:
           a cov_pop_burden_df with target_pop column added
    """
    # Select incidence as the target population if it is a therapeutic or diagnostic
    if param_df.loc[index, 'intervention_type'] == 'Therapeutic':
        cov_pop_burden_df['target_pop'] = cov_pop_burden_df['incidence_number']
    elif param_df.loc[index, 'intervention_type'] == 'Therapeutic mental health':
        cov_pop_burden_df['target_pop'] = cov_pop_burden_df['incidence_number']
    elif param_df.loc[index, 'intervention_type'] == 'Diagnostic':
        cov_pop_burden_df['target_pop'] = cov_pop_burden_df['incidence_number']
    # Select population column if it is a vaccine 
    #~ This assumes it is an infant vaccination
    elif param_df.loc[index, 'intervention_type'] == 'Vaccine':
        cov_pop_burden_df['target_pop'] = cov_pop_burden_df['pop_0-0']
    else:
        raise ValueError('The value of intervention_type for is not valid')
    return cov_pop_burden_df
    
def apply_endemicity_threshold(cov_pop_burden_df, param_df, index):
    """Reduces coverage rates in df if burden is below the threshold
       Inputs:
           cov_pop_burden_df - a df with the column 'coverage'
           param_df - a df of parameters, must contain the columns coverage_below_threshold,
               endem_thresh and endem_thresh_metric
           index - a string that is one of the indexes of param_df
       Returns:
           a df with coverage reduced in geographies with burden below the threshold
    """
    # Defines which burden column is relevant for the endemicity threshold
    endem_thresh_column = param_df.loc[index, 'endem_thresh_metric']+'_rate'
    # Defines the threshold and coverage below the threshold
    endem_thresh = param_df.loc[index, 'endem_thresh']
    coverage_below_threshold = param_df.loc[index, 'coverage_below_threshold']
    # Applies the endemicity threshold
    cov_pop_burden_df['coverage'] = np.where(cov_pop_burden_df[endem_thresh_column] < endem_thresh,
                     coverage_below_threshold,
                     cov_pop_burden_df['coverage'])
    return cov_pop_burden_df

def apply_diagnostic_inflation(cov_pop_burden_df, param_df, index):
    """Inflates the target_pop in df 
       Inputs:
           cov_pop_burden_df - a df with the column 'target_pop'
           param_df - a df of parameters, must contain the columns 'inflation_factor'
           index - a string that is one of the indexes of param_df
       Returns:
           a df with inflated target_pop column
    """
    cov_pop_burden_df['target_pop'] = cov_pop_burden_df['target_pop']*param_df.loc[index, 'inflation_factor']
    return cov_pop_burden_df

def apply_intervention_cut(cov_pop_burden_df, param_df, index):
    """Reduces prob_cover by the intervention_cut
       Inputs:
           cov_pop_burden_df - a df with the column 'prob_cover'
           param_df - a df of parameters, must contain the columns 'intervention_cut'
           index - a string that is one of the indexes of param_df
       Returns:
           a df with inflated target_pop column
    """
    cov_pop_burden_df['prob_cover'] = cov_pop_burden_df['prob_cover']*param_df.loc[index, 'intervention_cut']
    return cov_pop_burden_df

def adjust_for_intervention_factors(cov_pop_burden_dict, param_dict):
    """Applies several adjustments to the dfs that are the values of the nested
       dict cov_pop_burden_dicts
       Inputs:
           cov_pop_burden_dict - a nested dict where keys are id_codes and then
               scenarios and the values are dfs 
           param_dict - a dict where the keys are id_codes and the values are dfs
               with a row of parameters for each of the scenarios
       Returns:
           a nested dict of updated dfs
    """
    for code in param_dict.keys():
        param_df = param_dict[code]
        for index in param_df.index.tolist():
            cov_pop_burden_df = cov_pop_burden_dict[code][index].copy()
            cov_pop_burden_df = create_target_population(cov_pop_burden_df, param_df, index)
            if param_df.loc[index, 'intervention_type'] == 'Diagnostic':
                cov_pop_burden_df = apply_diagnostic_inflation(cov_pop_burden_df, param_df, index)
            elif param_df.loc[index, 'intervention_type'] == 'Vaccine':
                cov_pop_burden_df = apply_endemicity_threshold(cov_pop_burden_df, param_df, index)
            cov_pop_burden_df = apply_intervention_cut(cov_pop_burden_df, param_df, index)
            cov_pop_burden_dict[code][index] = cov_pop_burden_df
    return cov_pop_burden_dict
    
def calculate_lives_touched(cov_pop_burden_df):
    """Creates a column for lives touched by multiplying relevant columns
       Inputs:
           cov_pop_burden_df - a df that must contain columns 'target_pop', 
               'coverage' and 'prob_cover'
       Returns:
           a df with a lives_touched column
    """
    cov_pop_burden_df['lives_touched'] = (cov_pop_burden_df['target_pop']*
                                          cov_pop_burden_df['coverage']*
                                          cov_pop_burden_df['prob_cover'])
    return cov_pop_burden_df

def update_lives_touched(cov_pop_burden_dict, param_dict):
    """Updates lives_touched figures for each of the scenarios in the param_dfs
       Inputs:
           cov_pop_burden_dict - a nested dict where keys are id_codes and then
               scenarios and the values are dfs
           param_dict - a dict where the keys are id_codes and the values are dfs
               with a row of parameters for each of the scenarios
       Returns:
           param_dict with updated lives_touched figures for each of the rows
               of the dfs (the values of the dict)
    """
    for code in param_dict.keys():
        param_df = param_dict[code]
        for index in param_df.index.tolist():
            #~haven't extracted the estimated lives_touched in the cov_pop_burden_dict, is it necessary?
            cov_pop_burden_df = cov_pop_burden_dict[code][index].copy()
            cov_pop_burden_df = calculate_lives_touched(cov_pop_burden_df)
            lives_touched = cov_pop_burden_df['lives_touched'].sum()
            param_df.loc[index, 'lives_touched'] = lives_touched
        param_dict[code] = param_df
    return param_dict
    
def calculate_lives_improved(param_df):
    """Create a new column for lives improved in a df
       Inputs:
           param_df - a df of parameters must contain columns: 'lives_touched' and 'efficacy'
       Returns:
           a df with a lives_improved column added
    """
    param_df['lives_improved'] = (param_df['lives_touched']*param_df['efficacy'])
    return param_df

def update_lives_improved(param_dict):
    """Update dfs in param_dict with lives improved columns
       Inputs:
           param_dict - a dict where keys are id_codes and the values are 
               dfs of parameters
       Returns:
           a dict of dfs with a lives improved column added for each of them
    """
    param_dict = {k: calculate_lives_improved(v) for k, v in param_dict.items()}
    return param_dict

def isolate_deterministic_rows(param_df):
    """Filter dfs by whether the index is text (signifying a deterministic scenario)
       Inputs:
           param_df - a df of paramters where the indexes are strings for 
               deterministic scenarios and numbers for probabilistic trials
       Returns:
           a filtered df where all the data is for deterministic scenarios
    """
    param_determ_df = param_df.loc[param_df.index.str.contains('[a-z]', na=False)].copy()
    return param_determ_df

def isolate_probabilistic_rows(param_df, analysis_type):
    """Filter dfs by whether the index is a number in the range 1:num_trials
       (signifying a probabilistic trial)
       Inputs:
           param_df - a df of paramters where the indexes are strings for 
               deterministic scenarios and numbers for probabilistic trials
       Returns:
           a filtered df where all the data is for probabilistic trials      
    """
    num_trials = analysis_type['num_trials']
    param_prob_df = param_df.loc[param_df.index.isin(range(1,num_trials+1))].copy()
    return param_prob_df

def separate_param_dict(param_dict, analysis_type):
    """Creates a dictionary of two dictionaries, one for deterministic scenarios
       and another for probabilistic trials
       Inputs:
           param_dict - a dict where keys are id_codes and values are dfs
       Returns:
           a dict of two dicts 
    """
    deterministic_dict = {k: isolate_deterministic_rows(v) for k, v in param_dict.items()}
    probabilistic_dict = {k: isolate_probabilistic_rows(v, analysis_type) for k, v in param_dict.items()}
    return {'det': deterministic_dict, 'prob': probabilistic_dict}
 
def tornado_matplotlib(graph_data, base, directory, file_name):
    """Creates a tornado diagram and saves it to a prespecified directory
       Inputs:
           graphs_data - a df which must contain the columns 'variables' for 
               the names of the variables being graphed, 'lower' for the lower
               bounds of the deterministic sensitivity and 'ranges' for the total
               range between the lower and upper
           base - a float with the base case value
           directory - str - a path to a directory where the plot should be saved
           file_name - str- a useful name by which the plot with be saved
       Exports:
           A chart to the prespecified directory
    """
    # Sort the graph data so the widest range is at the top and reindex
    graph_data.copy()
    graph_data = graph_data.sort_values('ranges', ascending = False)
    graph_data.index = list(range(len(graph_data.index)))[::-1]

    # The actual drawing part
    fig = plt.figure()
    
    # Plot the bars, one by one
    for index in graph_data.index:
        # The width of the 'low' and 'high' pieces
        low = graph_data.loc[index, 'lower']
        value = graph_data.loc[index, 'ranges']
        low_width = base - low
        high_width = low + value - base
    
        # Each bar is a "broken" horizontal bar chart
        plt.broken_barh(
            [(low, low_width), (base, high_width)],
            (index - 0.4, 0.8),
            facecolors=['red', 'green'],  # Try different colors if you like
            edgecolors=['black', 'black'],
            linewidth=1,
        )
    
    # Draw a vertical line down the middle
    plt.axvline(base, color='black', linestyle='dashed')
    
    # Position the x-axis and hide unnecessary axes
    ax = plt.gca()  # (gca = get current axes)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    
    # Make the y-axis display the variables
    plt.yticks(graph_data.index.tolist(), graph_data['variables'])
    
    # Set the portion of the x- and y-axes to show
    plt.xlim(left = 0)
    plt.ylim(-1, len(graph_data.index))
    
    # Stop scientific formats
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    fig.autofmt_xdate()
    
    # Change axis text format
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.family': 'calibri'})
    
    # Make it tight format 
    plt.tight_layout()
    
    # Set up export path and export the chart
    path = directory+file_name+'.png'
    fig.savefig(path)
    print('Please find the chart at', path)
    plt.close(fig=None)

def probability_histogram(graph_data, variable, directory, file_name):
    """Creates a probability histogram and exports it to a directory
       Inputs:
           graph_data - df 
           variable - str - the variable in graph_data to be histogrammed
           directory - str - a path to a directory where the plot should be saved
           file_name - str- a useful name by which the plot with be saved
       Exports:
           A chart to the prespecified directory
    """
    # Get the right series and make it np
    graph_data = graph_data[variable]
    
    # Calculate confidence intervals
    upper = graph_data.quantile(0.975)
    lower = graph_data.quantile(0.025)
    # Histogram:
    # Bin it
    n, bin_edges = np.histogram(graph_data, 30)
    # Normalize it, so that every bins value gives the probability of that bin
    bin_probability = n/float(n.sum())
    # Get the mid points of every bin
    bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
    # Compute the bin-width
    bin_width = bin_edges[1]-bin_edges[0]
    # Plot the histogram as a bar plot
    fig, ax = plt.subplots()
    plt.bar(bin_middles, bin_probability, width=bin_width)
    
    # Add 95% confidence interval lines
    plt.axvline(upper, color='black', linestyle='dashed')
    plt.axvline(lower, color='black', linestyle='dashed')
    
    # Axis labels
    label = re.sub('_', ' ', variable).title()
    plt.xlabel(label)
    plt.ylabel('Probability')
    
    # Set the portion of the x- and y-axes to show
    plt.xlim(left = 0)
    
    # Stop axis scientific formats
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    fig.autofmt_xdate()
    
    # Change axis text format
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.family': 'calibri'})
    
    # Make it tight format 
    plt.tight_layout()
    
    # Export
    path = directory+file_name+'.png'    
    plt.savefig(path)
    print('Please find the chart at', path)
    plt.close(fig=None)

def restructure_graph_data_deterministic(param_df, variable):
    """Restructures data to the form needed by the tornado diagram
       Inputs:
           param_df - a df that must have indexes representing the different
               deterministic scenarios being modelled
           variable - str - the name of the variable to be plotted on the x axis
               normally either lives touched or lives improved
       Returns:
           base - float -  the base case estimate for the variable
           graph_data - a df in the format needed for tornado_matplotlib()
    """
    base = param_df.loc['base', variable]
    lower_series = param_df[param_df.index.str.contains('lower')][variable]
    lower_series.name = 'lower'
    lower_series.index = [re.sub('_lower', '', index) 
                          for index in lower_series.index]
    upper_series = param_df[param_df.index.str.contains('upper')][variable]
    upper_series.name = 'upper'
    upper_series.index = [re.sub('_upper', '', index) 
                          for index in upper_series.index]
    graph_data = pd.concat([upper_series, lower_series], axis = 1)
    graph_data['ranges'] = abs(graph_data['upper'] - graph_data['lower'])
    graph_data['variables'] = graph_data.index
    graph_data = graph_data[graph_data['ranges']>0]
    graph_data.index = range(len(graph_data.index))
    return base, graph_data

def draw_graphs_export(probabilistic_dict, deterministic_dict, directory):
    """Exports graphs to a given directory based on the data it is given
           Inputs:
               probabilistic_dict - dict - keys are codes and values are dfs
                   of trial parameters and estimates
               deterministic_dict - dict - keys are codes and values are dfs
                   of scenario parameters and estimates
               directory - str - the directory where graphs should be exported
           Exports:
               4 graphs for each code to the directory
    """
    for code in probabilistic_dict.keys():
        graph_data = probabilistic_dict[code]
        variable = 'lives_touched'
        file_name = code+'_probabilistic_'+variable
        probability_histogram(graph_data, variable, directory, file_name)
        variable = 'lives_improved'
        file_name = code+'_probabilistic_'+variable
        probability_histogram(graph_data, variable, directory, file_name)
    
    for code in deterministic_dict.keys():
        param_df = deterministic_dict[code]
        variable = 'lives_touched'
        base, graph_data = restructure_graph_data_deterministic(param_df, variable)
        file_name = code+'_deterministic_'+variable
        tornado_matplotlib(graph_data, base, directory, file_name)
        variable = 'lives_improved'
        base, graph_data = restructure_graph_data_deterministic(param_df, variable)
        file_name = code+'_deterministic_'+variable
        tornado_matplotlib(graph_data, base, directory, file_name)

def update_param_user_all(deterministic_dict, probabilistic_dict, param_user_all, param_user):
    """Puts the new values for lives_touched and lives_improved (including the
       95% confidence interval upper and lower bounds)
       Inputs:
           deterministic_dict - a dictionary where keys are id_codes and values
               are dfs where one of the indexes is 'base' and columns are 
               'lives_touched' and 'lives_improved'
           probabilistic_dict - a dictionary where keys are id_codes and values
               are dfs where columns are 'lives_touched' and 'lives_improved'
           param_user_all - a df where indexes are id_codes 
    """
    for code in param_user.keys():
        determ_df = deterministic_dict[code]
        prob_df = probabilistic_dict[code]
        param_user_all.loc[code, 'lives_touched'] = determ_df.loc['base', 'lives_touched']
        param_user_all.loc[code, 'lives_improved'] = determ_df.loc['base', 'lives_improved']
        param_user_all.loc[code, 'lives_touched_025'] = prob_df['lives_touched'].quantile(0.025)
        param_user_all.loc[code, 'lives_touched_975'] = prob_df['lives_touched'].quantile(0.975)
        param_user_all.loc[code, 'lives_improved_025'] = prob_df['lives_improved'].quantile(0.025)
        param_user_all.loc[code, 'lives_improved_975'] = prob_df['lives_improved'].quantile(0.975)
    return param_user_all

def export_estimates(param_user_all, analysis_type):
    """Overwrite the main csv with the updated version of estimates / paramters
       Inputs:
           param_user_all - a df of parameters and estimates
       Writes:
           csvs to two different locations
    """
    # Write back up csv
    time_now = datetime.datetime.now()
    data_str = time_now.strftime('%Y_%m_%d_%H_%M')
    back_up_path = 'C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/data/backup/LTLI_parameters_python_'+data_str+'.csv'
    param_user_all.to_csv(back_up_path)
    # Overwrite imported csv
    if analysis_type['overwrite_parameters']:
        param_user_all.to_csv('LTLI_parameters_python.csv')

# Code run sequentially
def main():
    pass

if __name__ == "__main__":
   
    # Write in a parameter checking function as the first function
    check_inputs(analysis_type, param_user_all, population, coverage, burden_all)
    
    # Vary the parameter dict depending on whether you are running all the analysis
    # or just a subset
    param_user = check_run_all(analysis_type, param_user_dict)
    
    # Create different versions of the parameters ready for sensitivity analyses
    deterministic_dict = restructure_to_deterministic(analysis_type, param_user)
    probabilistic_dict = restructure_to_probabilistic(analysis_type, param_user)
    
    # Combine into one dict
    param_dict = {k: pd.concat([deterministic_dict[k], probabilistic_dict[k]])
                  for k in deterministic_dict.keys()}
    
    #~ SHOULD PROBABLY ADD IN A FUNCTION HERE TO CLEAR OUT THE VALUES OF LTLI TO AVOID POTENTIAL CONFUSION
    
    # Get the disease burden for each disease
    burden_dict = get_relevant_burden(param_dict, burden_all)
    
    # Adjust burden so it is in a dictionary of relevant burden dfs for each trial
    # respectively
    burden_dict = adjust_burden(burden_dict, param_dict)
    
    # Create the cov_pop_dict based on coverage for that type of intervention an
    # population
    cov_pop_dict = create_coverage_population_dict(coverage, population, param_dict)
    
    # Adjust cov_pop_dict so the values are now dicts where the keys are scenarios
    # and the values are dfs customised to the scenarios
    cov_pop_dict = adjust_cov_pop_for_trials(cov_pop_dict, param_dict)
    
    # Merge the burden and coverage / population dfs
    cov_pop_burden_dict = merge_cov_pop_and_burden(burden_dict, cov_pop_dict)
    
    # Adjust the dfs in cov_pop_burden_dict for intervention factors such as the 
    # endemicity threshold, diagnostic inflation and intervention cut
    cov_pop_burden_dict = adjust_for_intervention_factors(cov_pop_burden_dict, param_dict)
    
    # Calculate lives_touched and input them to 
    param_dict = update_lives_touched(cov_pop_burden_dict, param_dict)
    
    param_dict = update_lives_improved(param_dict)
    
    # Separate param dict into seperate dicts for further analyis
    param_dict_separated = separate_param_dict(param_dict, analysis_type)
    deterministic_dict = param_dict_separated['det']
    probabilistic_dict = param_dict_separated['prob']
       
    # Export graphs for all of the analyses
    draw_graphs_export(probabilistic_dict, deterministic_dict, directory)
    
    # Update param_user_all ready for export
    param_user_all = update_param_user_all(deterministic_dict, 
                                           probabilistic_dict, 
                                           param_user_all,
                                           param_user)
    export_estimates(param_user_all, analysis_type)
    
    main()

#~NEED TO FIND THE COVERAGE DATA nan I SAW?????
    
# Write in corrections so therapeutic mental health can pull in the right coverage buckets