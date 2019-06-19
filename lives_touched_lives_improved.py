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
    
    # Clear any previous exception comments for projects that are being modelled
    estimates_output = apply_exceptions.clear_exceptions(param_user_all, param_user)
    
    # Apply various geographical exceptions see the updated
    cov_pop_burden_dict = apply_exceptions.apply_geography_exceptions(cov_pop_burden_dict, 
                                                                      estimates_output)
    
    # Calculate lives_touched and input them to 
    param_dict = calculate_ltli.update_lives_touched(cov_pop_burden_dict, param_dict)
    
    param_dict = calculate_ltli.update_lives_improved(param_dict)
    
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
    
    # Draws graphs and exports them to graphs_dir for all of the analyses
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

