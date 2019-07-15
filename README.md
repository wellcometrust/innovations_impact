# Lives touched, lives improved model

This repository is for Innovations lives touched, lives improved model, which seeks to estimate the potential impact of interventions were they to be successfully developed. This model relies on assumptions abstract in order to model scenarios of potential impact - this results in highly uncertain estimates which is quantified.

## How to setup:

This repository requires Python 3.6+.

The required files can be found [here](https://wellcomecloud.sharepoint.com/sites/innovations/PAD/Forms/AllItems.aspx?id=%2Fsites%2Finnovations%2FPAD%2FPAD%20Archive%2FData%20Analyst%20Handover%2Flives%5Ftouched%5Flives%5Fimproved).

The required packages are set out in requirements.txt.

You also need to sync this [folder](https://wellcomecloud.sharepoint.com/sites/innovations/PAD/Forms/AllItems.aspx?id=%2Fsites%2Finnovations%2FPAD%2FLives%20touched%2C%20lives%20improved%20model%20results) to OneDrive to be able to upload the model charts to the right place.

More detailed instructions on how to set up your computer and run the model can be found [here](https://wellcomecloud.sharepoint.com/:p:/r/sites/innovations/PAD/PAD%20Archive/Data%20Analyst%20Handover/lives_touched_lives_improved_support/Instruction%20for%20updating%20LTLI%20V2.pptm?d=we7c4a802284d48528e301f0a36865c16&csf=1&e=0H6OI9).

## How to run:

lives_touched_lives_improved.py is the main file to run the model from (it calls on the other modules in the 
repository). 

It can be called from the command line and arguments passed to it to indicate the directories and file names where the required files can be found.

It can be also run from an IDE - at a minimum the file names and directory paths will need updating - see more detailed instructions on that [here](https://wellcomecloud.sharepoint.com/:p:/r/sites/innovations/PAD/PAD%20Archive/Data%20Analyst%20Handover/lives_touched_lives_improved_support/Instruction%20for%20updating%20LTLI%20V2.pptm?d=we7c4a802284d48528e301f0a36865c16&csf=1&e=0H6OI9).

To change the default behaviour of the lives_touched_lives_improved.py change the default terms in ANALYSIS_TYPE.

If ANALYSIS_TYPE is set 'run_all': False in lives_touched_lives_improved.py then you will have to input a valid id_code to run your chosen model. The id_codes come from LTLI_paramters.csv - check there to see which code you want to run.

## How to model new projects:

All of the model parameters are contained in LTLI_parameters.csv. To model a new project then simply add a row to the csv with the new parameters. 

Find instructions on how to input valid and sensible parameters [here](https://wellcomecloud.sharepoint.com/:p:/r/sites/innovations/PAD/PAD%20Archive/Data%20Analyst%20Handover/lives_touched_lives_improved_support/Instruction%20for%20updating%20LTLI%20V2.pptm?d=we7c4a802284d48528e301f0a36865c16&csf=1&e=0H6OI9). 

## Note carefully:

1. There are not tests associated with all of the functions in this package.
2. There are input_checks which will throw errors if the datasets do not have the required columns, if in doubt then follow the format of the files handed over [here](https://wellcomecloud.sharepoint.com/sites/innovations/PAD/Forms/AllItems.aspx?id=%2Fsites%2Finnovations%2FPAD%2FPAD%20Archive%2FData%20Analyst%20Handover%2Flives%5Ftouched%5Flives%5Fimproved) (column names etc.).
