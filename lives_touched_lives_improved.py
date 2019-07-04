# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 08:48:52 2019

Lives touched, lives improved model

@author: LaurencT
"""

import pandas as pd
from argparse import ArgumentParser
import sys

# Import other modules written for LTLI
sys.path.append('C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/scripts')
import model_inputs
import input_check
from generate_deterministic_params import get_deterministic_params
from generate_probabilistic_params import get_probabilistic_params
from restructure_cov_pop_burden import get_relevant_burden
from restructure_cov_pop_burden import adjust_burden_dict
from restructure_cov_pop_burden import create_coverage_population_dict
from restructure_cov_pop_burden import adjust_cov_pop_for_trials
from restructure_cov_pop_burden import merge_cov_pop_and_burden
from adjust_for_intervention_factors import adjust_for_intervention_factors
import apply_exceptions
import calculate_ltli
import reshape_for_graphs
import exports

# Set up directories, file names and analysis type
DATA_DIR = 'C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/data/'
GRAPH_DIR = 'C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/graphs/'
OUTPUTS_DIR = 'C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/outputs/'
BACKUP_DIR = 'C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/data/backup/'
SLIDES_DIR = 'C:/Users/laurenct/Wellcome Cloud/Innovations - Lives touched, lives improved model results/'

PARAM_CSV_NAME = 'LTLI_parameters.csv'
ESTIMATES_CSV_NAME = 'LTLI_outputs.csv'
POPULATION_CSV_NAME = 'GBD_population_2016_reshaped.csv'
BURDEN_CSV_NAME = 'gbd_data_wide_2017.csv'
COVERAGE_XLS_NAME = 'intervention_coverage_assumptions.xlsm'
COVERAGE_SHEET_NAME = 'Penetration assumptions'
PPT_TEMPLATE_NAME = 'mm_template_impact.pptx'

ANALYSIS_TYPE = {'run_all' : False, # True if you just want to run all the models or False to run one model 
                 'num_trials' : 1000, # 1000 as standard - could do fewer to speed up
                 'overwrite_estimates' : True # True or False - do you want to overwrite the estimates linked to Tableau
                  }

parser = ArgumentParser(description = 'Input strings to override the default ' +
                        'folder locations required to get data and write outputs')

parser.add_argument('--DATA_DIR', 
                    type = str, 
                    help = 'A string indicating the folder containing the' +
                           ' model data. The default is ' + DATA_DIR, 
                    default = DATA_DIR,
                    required = False)
parser.add_argument('--GRAPH_DIR', 
                    type = str, 
                    help = 'A string indicating the folder where the graphs' +
                           ' should be exported. The default is ' + GRAPH_DIR, 
                    default = GRAPH_DIR,
                    required = False)
parser.add_argument('--OUTPUTS_DIR', 
                    type = str, 
                    help = 'A string indicating the folder where the outputs' +
                           ' should be exported. The default is ' + OUTPUTS_DIR, 
                    default = OUTPUTS_DIR,
                    required = False)
parser.add_argument('--BACKUP_DIR', 
                    type = str, 
                    help = 'A string indicating the folder where the parameters' +
                           ' should be backed up. The default is ' + BACKUP_DIR, 
                    default = BACKUP_DIR,
                    required = False)
parser.add_argument('--SLIDES_DIR', 
                    type = str, 
                    help = 'A string indicating the folder where the ppt slides' +
                           ' should be exported. The default is ' + SLIDES_DIR, 
                    default = SLIDES_DIR,
                    required = False)
parser.add_argument('--PARAM_CSV_NAME', 
                    type = str, 
                    help = 'A string indicating the file name for the model' +
                           ' parameters. The default is ' + PARAM_CSV_NAME, 
                    default = PARAM_CSV_NAME,
                    required = False)
parser.add_argument('--ESTIMATES_CSV_NAME', 
                    type = str, 
                    help = 'A string indicating the file name for the output' +
                           ' estimates. The default is ' + ESTIMATES_CSV_NAME, 
                    default = ESTIMATES_CSV_NAME,
                    required = False)
parser.add_argument('--POPULATION_CSV_NAME', 
                    type = str, 
                    help = 'A string indicating the file name for the population' +
                           ' data. The default is ' + POPULATION_CSV_NAME, 
                    default = POPULATION_CSV_NAME,
                    required = False)
parser.add_argument('--BURDEN_CSV_NAME', 
                    type = str, 
                    help = 'A string indicating the file name for the burden' +
                           ' data. The default is ' + BURDEN_CSV_NAME, 
                    default = BURDEN_CSV_NAME,
                    required = False)
parser.add_argument('--COVERAGE_XLS_NAME', 
                    type = str, 
                    help = 'A string indicating the file name for the coverage' +
                           ' data. The default is ' + COVERAGE_XLS_NAME, 
                    default = COVERAGE_XLS_NAME,
                    required = False)
parser.add_argument('--COVERAGE_SHEET_NAME', 
                    type = str, 
                    help = 'A string indicating the excel sheet name for the' +
                           ' coverage data. The default is ' + COVERAGE_SHEET_NAME, 
                    default = COVERAGE_SHEET_NAME,
                    required = False)
parser.add_argument('--PPT_TEMPLATE_NAME', 
                    type = str, 
                    help = 'A string indicating the file name for the ppt template ' +
                           ' to use for the slides. The default is ' + PPT_TEMPLATE_NAME, 
                    default = PPT_TEMPLATE_NAME,
                    required = False)

args = parser.parse_args()

DATA_DIR = args.DATA_DIR
GRAPH_DIR = args.GRAPH_DIR
OUTPUTS_DIR = args.OUTPUTS_DIR
BACKUP_DIR = args.BACKUP_DIR
SLIDES_DIR = args.SLIDES_DIR
PARAM_CSV_NAME = args.PARAM_CSV_NAME
ESTIMATES_CSV_NAME = args.ESTIMATES_CSV_NAME
POPULATION_CSV_NAME = args.POPULATION_CSV_NAME
BURDEN_CSV_NAME = args.BURDEN_CSV_NAME
COVERAGE_XLS_NAME = args.COVERAGE_XLS_NAME
COVERAGE_SHEET_NAME = args.COVERAGE_SHEET_NAME
PPT_TEMPLATE_NAME = args.PPT_TEMPLATE_NAME

print('You selected', DATA_DIR, 'as the DATA_DIR', sep = ' ')
print('You selected', GRAPH_DIR, 'as the GRAPH_DIR', sep = ' ')
print('You selected', OUTPUTS_DIR, 'as the OUTPUTS_DIR', sep = ' ')
print('You selected', BACKUP_DIR, 'as the BACKUP_DIR', sep = ' ')
print('You selected', SLIDES_DIR, 'as the SLIDES_DIR', sep = ' ')
print('You selected', PARAM_CSV_NAME, 'as the PARAM_CSV_NAME', sep = ' ')
print('You selected', ESTIMATES_CSV_NAME, 'as the ESTIMATES_CSV_NAME', sep = ' ')
print('You selected', POPULATION_CSV_NAME, 'as the POPULATION_CSV_NAME', sep = ' ')
print('You selected', BURDEN_CSV_NAME, 'as the BURDEN_CSV_NAME', sep = ' ')
print('You selected', COVERAGE_XLS_NAME, 'as the COVERAGE_XLS_NAME', sep = ' ')
print('You selected', COVERAGE_SHEET_NAME, 'as the COVERAGE_SHEET_NAME', sep = ' ')
print('You selected', PPT_TEMPLATE_NAME, 'as the PPT_TEMPLATE_NAME', sep = ' ')

# Code run sequentially
if __name__ == "__main__":
   
    # Loads the relevant model parameters
    param_user_all = model_inputs.load_params(PARAM_CSV_NAME, DATA_DIR)
    estimates_output = model_inputs.load_params(ESTIMATES_CSV_NAME, OUTPUTS_DIR)
    
    
    # Transforms the parameters to a dict for future transformation
    param_user_dict = model_inputs.create_param_dict(param_user_all)
    
    population = model_inputs.load_population_data(POPULATION_CSV_NAME, DATA_DIR) 
    
    burden_all = model_inputs.load_burden_data(BURDEN_CSV_NAME, DATA_DIR)
    
    coverage = model_inputs.load_coverage_assumptions(COVERAGE_XLS_NAME,
                                                      COVERAGE_SHEET_NAME,
                                                      DATA_DIR)

    # Write in a parameter checking function as the first function
    input_check.check_inputs(ANALYSIS_TYPE, 
                             param_user_all, 
                             population, 
                             coverage, 
                             burden_all)
    
    # Vary the parameter dict depending on whether you are running all the analysis
    # or just a subset
    param_user = input_check.check_run_all(ANALYSIS_TYPE, 
                                           param_user_dict)
        
    # Create different versions of the parameters ready for sensitivity analyses
    deterministic_dict = get_deterministic_params(ANALYSIS_TYPE, param_user)
    probabilistic_dict = get_probabilistic_params(ANALYSIS_TYPE, param_user)
    
    # Combine all parameters into one dict
    param_dict = {k: pd.concat([deterministic_dict[k], probabilistic_dict[k]])
                  for k in deterministic_dict.keys()}
    
    print('All the parameters have been simulated')
        
    # Get the disease burden data for each set of parameters
    burden_dict_unadjusted = get_relevant_burden(param_dict, burden_all)
    
    # Adjust burden so it is in a dictionary of relevant burden dfs for each set
    # of parameters
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
    
    print('All the datasets have been simulated based on the parameters')
    
    # Adjust the dfs in cov_pop_burden_dict for intervention factors such as the 
    # endemicity threshold, diagnostic inflation and intervention cut
    cov_pop_burden_dict = adjust_for_intervention_factors(cov_pop_burden_dict, 
                                                          param_dict)
    
    # Clear any previous exception comments for projects that are being modelled
    estimates_output = apply_exceptions.clear_exceptions(estimates_output, param_user)
    
    # Apply various geographical exceptions see the updated
    cov_pop_burden_dict = apply_exceptions.apply_geography_exceptions(cov_pop_burden_dict, 
                                                                      estimates_output)
    
    # Calculate lives_touched and input them to 
    param_dict = calculate_ltli.update_lives_touched(cov_pop_burden_dict, param_dict)
    
    param_dict = calculate_ltli.update_lives_improved(param_dict)
    
    print('All the estimates have been completed')
    
    # Separate param dict into seperate dicts for further analyis
    param_dict_separated = reshape_for_graphs.separate_param_dict(param_dict, ANALYSIS_TYPE)
    deterministic_dict = param_dict_separated['det']
    probabilistic_dict = param_dict_separated['prob']
    
    # Create base dicts for bridging diagram          
    base_cov_pop_burden_dict = {k: cov_pop_burden_dict[k]['base']
                                for k in cov_pop_burden_dict.keys()}
    
    bridge_graph_dict = reshape_for_graphs.get_bridging_data(base_cov_pop_burden_dict, 
                                                             burden_dict_unadjusted,
                                                             param_user_dict)
    
    # Draws graphs and exports them to graphs_dir for all of the analyses
    exports.draw_graphs_export(probabilistic_dict,
                       deterministic_dict, 
                       bridge_graph_dict, 
                       GRAPH_DIR)
    
    # Turn the graphs into formatted slides
    exports.create_all_slides(param_dict, 
                      SLIDES_DIR, 
                      GRAPH_DIR, 
                      PPT_TEMPLATE_NAME)
    
    # Update param_user_all ready for export
    estimates_output = exports.update_estimates_output(deterministic_dict, 
                                                       probabilistic_dict, 
                                                       estimates_output,
                                                       param_user)
    
    exports.export_estimates(estimates_output, 
                             ANALYSIS_TYPE, 
                             BACKUP_DIR, 
                             OUTPUTS_DIR, 
                             ESTIMATES_CSV_NAME)

    print('All the exports have been completed, the entire process is complete')