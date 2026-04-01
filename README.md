# Project Overview
Purpose of this code is to run tests of the coffea code on samples produced for the 2b2tau studies and for the 4tau analysis.
Selections applied will be both those used in the 2b2tau analysis and the 4tau analysis depending on the study of interest.
All code should be run on the singularity shells. Shell scripts to set up the singularity shells are included in each of the directories of interest.
The directory sturcture is meant to allow multiple different studies to be done efficienctly in parallel keeping plots produced with different selections seperated to avoid confusion.
The naming of directories/subdirectories is intended to document what selections are applied to all results (plots, coffea files, etc.) within the directory/subdirectory of interest.

## Running 4tau vs 2b2tau Studies
The finalized code comparing the 4tau skims with the 2b2tau skims is located in `Studies_4tau/UnifiedProcessor_Dir`. As of writing this code is only known to work with regularity on the Wisconsin Analysis Facility using the image `coffea-base-almalinux9:0.7.30-py3.10`. 

To produce the 4tau plots navigate to the directory `Studies_4tau/UnifiedProcessor_Dir` in a terminal on the AF and run the script `Comp_Script_4tau_Samples.py`, once the file completes running naviagte to the directory `Output_4Tau` and generate plots by running the command `python3 Make_Coffea_Plots.py -f 'name_of_coffea_file' -n 4`. 

To produce the 2b2tau plots navigate to the directory `Studies_4tau/UnifiedProcessor_Dir` in a terminal on the AF and run the script `Comp_Script_2b2tau_Samples.py`, once the file completes running naviagte to the directory `Output_2b2Tau` and generate plots by running the command `python3 Make_Coffea_Plots.py -f 'name_of_coffea_file' -n 4`. 
