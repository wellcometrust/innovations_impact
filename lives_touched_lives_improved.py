# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 08:48:52 2019

Lives touched, lives improved model

@author: LaurencT
"""
# Section 1 (see README for context)
# Import all standardised libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import beta
import re
import datetime
from pptx import Presentation
import sys
import os

# Import other modules written for LTLI
sys.path.append('C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/scripts')
import model_inputs
import input_check
from probability_distribution_moments import beta_moments
from probability_distribution_moments import gamma_moments
from probability_distribution_moments import gamma_moments_burden
import apply_exceptions
import graph
import reshape_for_graphs
import exports

# Set up directories, file names and analysis type
data_dir = 'C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/data/'
graph_dir = 'C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/graphs/'
outputs_dir = 'C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/outputs/'
backup_dir = 'C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/data/backup/'
slides_dir = 'C:/Users/laurenct/Wellcome Cloud/Innovations - Lives touched, lives improved model results/'

param_csv_name = 'LTLI_parameters.csv'
estimates_csv_name = 'LTLI_outputs.csv'
population_csv_name = 'GBD_population_2016_reshaped.csv'
burden_csv_name = 'gbd_data_wide_2017.csv'
coverage_xls_name = 'intervention_coverage_assumptions.xlsm'
coverage_sheet_name = 'Penetration assumptions'
ppt_template_name = 'mm_template_impact.pptx'

analysis_type = {'run_all' : False, # True or False
                 'run_deterministic' : True, # True or False TODO write in functionality to stop it running
                 'run_probabilistic' : True, # True or False TODO write in functionality to stop it running
                 'num_trials' : 1000, # 1000 as standard - could do 100 to speed up
                 'overwrite_estimates' : True # True or False
                  } 

# Section 3:
# Importing a csv of saved paramters and setting the id to be the index



# Section 4
# Declaring all the functions used in the script

def clear_exceptions(estimates_output, param_dict):
    """Clears exceptions column of estimates_output df for all models that are 
       going to be run (this is needed because the str defining exceptions get 
       concatenated)
    """
    estimates_output = estimates_output.copy()
    for code in param_dict.keys():
        estimates_output.loc[code, 'exception_count'] = 0
        estimates_output.loc[code, 'exception_comment'] = '.'
    return estimates_output

# Section 5:
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
    """Defines the different LTLI scenarios and which list of parameters should
       be used for each of them
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
               being undertaken, must contain the key 'run_deterministic'
           param_user - dict - keys: id_codes, values: series of parameters
       Returns:
           A dict - keys: scenario names, values - the list of parameters to
               be called for the corresponding scenario. Example:
               {'base': [burden_mean,
                         ...,
                         coverage_mean],
                '....': [...],
                'burden_lower': [burden_lower,
                                 ...,
                                 coverage_mean]}
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
            scenario_type = scenario[scenario.rindex('_')+1:]
            new_columns = [re.sub('mean', scenario_type, column) 
                           if re.search('disease.*prop', column) else column 
                           for column in columns_to_select]
        # All others replace the column index for mean with upper or lower
        else:
            scenario_focus = scenario[:scenario.rindex('_')] 
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

def get_deterministic_params(analysis_type, param_user):
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
        param_df = pd.concat(scenario_params, axis = 1, 
                             join_axes=[scenario_params[0].index])
        param_df = param_df.transpose()
        # populate the dictionary
        param_dict[code] = param_df
    return param_dict

# Section 6:
def get_probabilistic_column_names(param_user):
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

def simulate_column(mean, sd, column, num_trials):
    """Simulates num_trials values of a parameter based on the mean, sd and column type
       Inputs:
           mean - float - the base case value of that parameter
           sd - float - the standard deviation of that parameter
           column - str - the name of column to be  simulated
           num_trials - int - the num of trials to be simulated
       Returns:
           a numpy array of simulated values
    """
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
        # Disease prop may not be a proportion - may inflate the naive population
        if (re.search('disease_[1-3]_prop', column) and mean>1):
            data = norm.rvs(size = num_trials, loc = mean, scale = sd)
        else:
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
    return data

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
        data = simulate_column(mean, sd, column, num_trials)
        # Turn the relevant new data into a series (which becomes a column)
        new_column = pd.Series(data, index = range(1, num_trials + 1), name = column)
        prob_df = pd.concat([prob_df, new_column.T], axis = 1, sort = False)
        # Drop unnecessary columns
        columns_to_drop = [name for name in list(prob_df) if re.search("mean|SD", name)]
        prob_df = prob_df.drop(columns_to_drop, axis =1)
    return prob_df
    
def get_probabilistic_params(analysis_type, param_user):
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
    column_names = get_probabilistic_column_names(param_user)
    new_columns = columns_to_add(column_names)
    param_prob = {k:v[column_names] for k, v in param_user.items()}
    # Generate the relevant id_codes
    id_codes = list(param_user.keys())
    param_dict = {code: create_prob_df(param_prob, code, new_columns, num_trials) 
                  for code in id_codes}
    return param_dict

# Section 7:
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
            sd = (
                    new_burden_df[root + '_mean'] -
                    new_burden_df[root + '_lower']
                 )/2
            prop_lower_mean = (new_burden_df[root + '_mean'] / 
                               new_burden_df[root + '_mean']).mean()
            gamma_vals = pd.DataFrame([gamma_moments_burden(mean_val, sd_val) 
                                       for mean_val, sd_val in zip(mean, sd)])
            new_burden_df[root + '_mean'] = gamma.rvs(a = gamma_vals['shape'], 
                                                      scale = gamma_vals['scale'], 
                                                      size = len(gamma_vals['shape']))
            new_burden_df[root + '_mean'] = new_burden_df[root + '_mean'] * \
                                            norm.rvs(1, (1-prop_lower_mean)/4)
            new_burden_df[root + '_mean'] = np.where(new_burden_df[root + '_mean'] <0, 
                                                     0, new_burden_df[root + '_mean'])
        except ValueError:
            if index == 'burden_lower':
                new_burden_df[root + '_mean'] = new_burden_df[root + '_lower']
            elif index == 'burden_upper':
                new_burden_df[root + '_mean'] = new_burden_df[root + '_upper']
        
    # Remove upper and lower columns as the relevant column is now the mean
    relevant_columns = [column for column in list(new_burden_df) 
                        if not re.search('upper|lower', column)]
    new_burden_df = new_burden_df[relevant_columns]
    # Create new column name mapping
    new_column_names_dict = {column: re.sub('_mean', '', column) 
                             for column in relevant_columns}  
    # Rename columns
    new_burden_df = new_burden_df.rename(columns = new_column_names_dict)
    return new_burden_df

def strain_adjust_burden_df(burden_df, param_df, index):
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
    column_list = [column for column in list(new_burden_df) 
                   if re.search('number|rate', column)]
    # Create a mapping of the disease and proportion
    disease_dict = {k: k+"_prop" for k in ['disease_1', 'disease_2', 'disease_3']}
    # Adjust the df for the substrain prop for each disease
    for column in column_list:
        for disease in disease_dict.keys():
            new_burden_df[column] = np.where(
                    new_burden_df['cause'] == param_df.loc[index, disease], 
                    new_burden_df[column]*param_df.loc[index, disease_dict[disease]], 
                    new_burden_df[column])
    return new_burden_df

def aggregate_burden_df(burden_df):
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
    summed_burden_df = new_burden_df.groupby(['country', 'age']).sum()
    # Turn the indexes back into columns in a df
    index_list = summed_burden_df.index.tolist()
    summed_df = pd.DataFrame(index_list, 
                             columns = ['country', 'age'], 
                             index = index_list)
    # Add the new cause name to the df
    summed_df['cause'] = new_cause_name
    # Merge the burden with the country, age group, and cause columns
    summed_df = pd.concat([summed_df, summed_burden_df], axis = 1)
    # Reindex so the country is the only index
    summed_df.index = [i[0] for i in summed_df.index.tolist()]
    # Merge in region / super region columns
    other_columns = new_burden_df[['country', 'super_region', 'region']].drop_duplicates()
    summed_df = summed_df.merge(other_columns, on = 'country')
    summed_df.index = summed_df['country']
    return summed_df
    
def adjust_burden_dict(burden_dict, param_dict):
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
            burden_df_index = strain_adjust_burden_df(burden_df_index, 
                                                      param_df, 
                                                      index)
            burden_df_index = aggregate_burden_df(burden_df_index)
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
        new_column_names = {column: re.sub(('vaccine_|diagnostic_|'
                                           'therapeutic_mental_health_|therapeutic_'), 
                                           '', column) 
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
    pop_columns = [column for column in list(new_cov_pop_df) 
                   if re.search('pop', column)]
    for column in pop_columns:
        new_cov_pop_df[column] = (new_cov_pop_df[column] *
                                  param_df.loc[index, 'population'])
    # Adjust coverage columns by the coverage assumption for this scenario
    cov_columns = [column for column in list(new_cov_pop_df) 
                   if re.search('coverage', column)]
    for column in cov_columns:
        new_cov_pop_df[column] = (new_cov_pop_df[column] * 
                                  param_df.loc[index, 'coverage'])
        new_cov_pop_df[column] = np.where(new_cov_pop_df[column] > 0.95, 
                                          0.95, new_cov_pop_df[column])
    # Adjust prob_cover columns by the coverage assumption for this scenario
    cov_columns = [column for column in list(new_cov_pop_df) 
                   if re.search('prob_cover', column)]
    for column in cov_columns:
        new_cov_pop_df[column] = (new_cov_pop_df[column] * 
                                  param_df.loc[index, 'prob_cover'])
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

# Section 8:
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
    if re.search('Therapeutic', param_df.loc[index, 'intervention_type']):
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
    new_cov_pop_burden_df = cov_pop_burden_df.copy()
    # Defines which burden column is relevant for the endemicity threshold
    endem_thresh_column = param_df.loc[index, 'endem_thresh_metric']+'_rate'
    # Defines the threshold and coverage below the threshold
    endem_thresh = param_df.loc[index, 'endem_thresh']
    coverage_below_threshold = param_df.loc[index, 'coverage_below_threshold']
    # Applies the endemicity threshold
    new_cov_pop_burden_df['coverage'] = (
            np.where(new_cov_pop_burden_df[endem_thresh_column] < endem_thresh,
            coverage_below_threshold,
            new_cov_pop_burden_df['coverage'])
                                        )
    return new_cov_pop_burden_df

def apply_diagnostic_inflation(cov_pop_burden_df, param_df, index):
    """Inflates the target_pop in df 
       Inputs:
           cov_pop_burden_df - a df with the column 'target_pop'
           param_df - a df of parameters, must contain the columns 'inflation_factor'
           index - a string that is one of the indexes of param_df
       Returns:
           a df with inflated target_pop column
    """
    new_cov_pop_burden_df = cov_pop_burden_df.copy()
    new_cov_pop_burden_df['target_pop'] = (new_cov_pop_burden_df['target_pop'] * 
                                           param_df.loc[index, 'inflation_factor'])
    return new_cov_pop_burden_df

def apply_intervention_cut(cov_pop_burden_df, param_df, index):
    """Reduces prob_cover by the intervention_cut
       Inputs:
           cov_pop_burden_df - a df with the column 'prob_cover'
           param_df - a df of parameters, must contain the columns 'intervention_cut'
           index - a string that is one of the indexes of param_df
       Returns:
           a df with inflated target_pop column
    """
    cov_pop_burden_df['prob_cover'] = (cov_pop_burden_df['prob_cover'] * 
                                       param_df.loc[index, 'intervention_cut'])
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
            cov_pop_burden_df = create_target_population(cov_pop_burden_df, 
                                                         param_df, 
                                                         index)
            if param_df.loc[index, 'intervention_type'] == 'Diagnostic':
                cov_pop_burden_df = apply_diagnostic_inflation(cov_pop_burden_df, 
                                                               param_df, 
                                                               index)
            elif param_df.loc[index, 'intervention_type'] == 'Vaccine':
                cov_pop_burden_df = apply_endemicity_threshold(cov_pop_burden_df, 
                                                               param_df, 
                                                               index)
            cov_pop_burden_df = apply_intervention_cut(cov_pop_burden_df, 
                                                       param_df, 
                                                       index)
            cov_pop_burden_dict[code][index] = cov_pop_burden_df
    return cov_pop_burden_dict

# Section 9:


# Section 10:
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

# Section 11:

def update_estimates_output(deterministic_dict, probabilistic_dict, estimates_output, param_user):
    """Updates estimates_output with new values for lives_touched and lives_improved 
       (including the 95% confidence interval upper and lower bounds)
       Inputs:
           deterministic_dict - a dictionary where keys are id_codes and values
               are dfs where one of the indexes is 'base' and columns are 
               'lives_touched' and 'lives_improved'
           probabilistic_dict - a dictionary where keys are id_codes and values
               are dfs where columns are 'lives_touched' and 'lives_improved'
           estimates_output - a df where indexes are id_codes 
           param_user - dict - keys are id_codes for only the models that have been run
       Returns:
           a df updated with estimates
    """
    for code in param_user.keys():
        determ_df = deterministic_dict[code]
        prob_df = probabilistic_dict[code]
        estimates_output.loc[code, 'lives_touched'] = determ_df.loc['base', 'lives_touched']
        estimates_output.loc[code, 'lives_improved'] = determ_df.loc['base', 'lives_improved']
        estimates_output.loc[code, 'lives_touched_025'] = prob_df['lives_touched'].quantile(0.025)
        estimates_output.loc[code, 'lives_touched_975'] = prob_df['lives_touched'].quantile(0.975)
        estimates_output.loc[code, 'lives_improved_025'] = prob_df['lives_improved'].quantile(0.025)
        estimates_output.loc[code, 'lives_improved_975'] = prob_df['lives_improved'].quantile(0.975)
    return estimates_output



# Section 15:    
# Code run sequentially
if __name__ == "__main__":
   
    # Loads the relevant model parameters
    param_user_all = model_inputs.load_params(param_csv_name, data_dir)
    estimates_output = model_inputs.load_params(estimates_csv_name, outputs_dir)
    
    
    # Transforms the parameters to a dict for future transformation
    param_user_dict = model_inputs.create_param_dict(param_user_all)
    
    population = model_inputs.load_population_data(population_csv_name, data_dir) 
    
    burden_all = model_inputs.load_burden_data(burden_csv_name, data_dir)
    
    coverage = model_inputs.load_coverage_assumptions(coverage_xls_name,
                                                      coverage_sheet_name,
                                                      data_dir)

    # Write in a parameter checking function as the first function
    input_check.check_inputs(analysis_type, 
                             param_user_all, 
                             population, 
                             coverage, 
                             burden_all)
    
    # Vary the parameter dict depending on whether you are running all the analysis
    # or just a subset
    param_user = input_check.check_run_all(analysis_type, 
                                           param_user_dict)
    
    # Clear any previous exception comments for projects that are being modelled
    estimates_output = clear_exceptions(param_user_all, param_user)
    
    # Create different versions of the parameters ready for sensitivity analyses
    deterministic_dict = get_deterministic_params(analysis_type, param_user)
    probabilistic_dict = get_probabilistic_params(analysis_type, param_user)
    
    # Combine into one dict
    param_dict = {k: pd.concat([deterministic_dict[k], probabilistic_dict[k]])
                  for k in deterministic_dict.keys()}
    
    #~ SHOULD PROBABLY ADD IN A FUNCTION HERE TO CLEAR OUT THE VALUES OF LTLI TO AVOID POTENTIAL CONFUSION
    
    # Get the disease burden for each disease
    burden_dict_unadjusted = get_relevant_burden(param_dict, burden_all)
    
    # Adjust burden so it is in a dictionary of relevant burden dfs for each trial
    # respectively
    burden_dict= adjust_burden_dict(burden_dict_unadjusted, param_dict)
    
    # Create the cov_pop_dict based on coverage for that type of intervention an
    # population
    cov_pop_dict = create_coverage_population_dict(coverage, 
                                                   population, 
                                                   param_dict)
    
    # Adjust cov_pop_dict so the values are now dicts where the keys are scenarios
    # and the values are dfs customised to the scenarios
    cov_pop_dict = adjust_cov_pop_for_trials(cov_pop_dict, 
                                             param_dict)
    
    # Merge the burden and coverage / population dfs
    cov_pop_burden_dict = merge_cov_pop_and_burden(burden_dict, 
                                                   cov_pop_dict)
    
    # Adjust the dfs in cov_pop_burden_dict for intervention factors such as the 
    # endemicity threshold, diagnostic inflation and intervention cut
    cov_pop_burden_dict = adjust_for_intervention_factors(cov_pop_burden_dict, 
                                                          param_dict)
    
    # Apply various geographical exceptions see the updated
    cov_pop_burden_dict = apply_exceptions.apply_geography_exceptions(cov_pop_burden_dict, 
                                                                      estimates_output)
    
    # Calculate lives_touched and input them to 
    param_dict = update_lives_touched(cov_pop_burden_dict, param_dict)
    
    param_dict = update_lives_improved(param_dict)
    
    # Separate param dict into seperate dicts for further analyis
    param_dict_separated = reshape_for_graphs.separate_param_dict(param_dict, analysis_type)
    deterministic_dict = param_dict_separated['det']
    probabilistic_dict = param_dict_separated['prob']
    
    # Create base dicts for bridging diagram          
    base_cov_pop_burden_dict = {k: cov_pop_burden_dict[k]['base']
                                for k in cov_pop_burden_dict.keys()}
    
    bridge_graph_dict = reshape_for_graphs.get_bridging_data(base_cov_pop_burden_dict, 
                                                             burden_dict_unadjusted,
                                                             param_user_dict)
    
    # Export graphs for all of the analyses
    exports.draw_graphs_export(probabilistic_dict,
                       deterministic_dict, 
                       bridge_graph_dict, 
                       graph_dir)
    
    # Turn the graphs into formatted slides
    exports.create_all_slides(param_dict, 
                      slides_dir, 
                      graph_dir, 
                      ppt_template_name)
    
    # Update param_user_all ready for export
    estimates_output = exports.update_estimates_output(deterministic_dict, 
                                           probabilistic_dict, 
                                           param_user_all,
                                           param_user)
    exports.export_estimates(estimates_output, 
                     analysis_type, 
                     backup_dir, 
                     outputs_dir, 
                     estimates_csv_name)
    