# Project Overview
Purpose of this code is to run tests of the coffea code on samples produced for the 2b2tau studies and for the 4tau analysis.
Selections applied will be both those used in the 2b2tau analysis and the 4tau analysis depending on the study of interest.
All code should be run on the singularity shells. Shell scripts to set up the singularity shells are included in each of the directories of interest.
The directory sturcture is meant to allow multiple different studies to be done efficienctly in parallel keeping plots produced with different selections seperated to avoid confusion.
The naming of directories/subdirectories is intended to document what selections are applied to all results (plots, coffea files, etc.) within the directory/subdirectory of interest.

## Running 4tau vs 2b2tau Studies
The finalized code comparing the 4tau skims with the 2b2tau skims is located in the directory `Studies_4tau/UnifiedProcessor_Dir`. 
As of writing this code is only known to work with regularity on the Wisconsin Analysis Facility (AF) using the image `coffea-base-almalinux9:0.7.30-py3.10`, the code can simply be run in a terminal instance on the AF.

To produce the 4tau plots navigate to the directory `Studies_4tau/UnifiedProcessor_Dir` in a terminal on the AF and run the script `Comp_Script_4tau_Samples.py`. 
Presently the output file will be named `output_4_boosted_tau_selec_SingleMuData_4TauSamples_WithSingleMuTrigger.coffea` and will be stored in the directory `Studies_4tau/UnifiedProcessor_Dir/Output_4Tau`.
Once the file completes running naviagte to the directory `Output_4Tau` and generate plots by running the command `python3 Make_Coffea_Plots.py -f 'name_of_coffea_file' -n 4`. 

To produce the 2b2tau plots navigate to the directory `Studies_4tau/UnifiedProcessor_Dir` in a terminal on the AF and run the script `Comp_Script_2b2tau_Samples.py`.
Presently the output file will be named `output_4_boosted_tau_selec_SingleMuData_2b2TauSamples_WithSingleMuTrigger.coffea` and will be stored in the directory `Studies_4tau/UnifiedProcessor_Dir/Output_2b2Tau`.
Once the file completes running naviagte to the directory `Output_2b2Tau` and generate plots by running the command `python3 Make_Coffea_Plots.py -f 'name_of_coffea_file' -n 4`. 

## Obtaining Number of Events Prior to Skimming and Sum of Gen Weights Prior to Skimming
There is a coffea processor that obtains the number of events prior to skimming and sum of gen weights prior to skimming included in this repository. 
The processor is called by a runner script (`PreSkimWeight_Runner.py`) that produces two json files with the number of events prior to skimming and sum of gen weights prior to skimming for all samples specified within `PreSkimWeight_Runner.py`. 
The output of the script is required when running studies on the 4 tau samples as the script `Comp_Script_4tau_Samples.py` requires a json file that contains the sum of gen weights or the sum of events prior to skimming.
The runner is currently hard coded to produce json files form 2018 UL samples.
Copies of these json files are already included in the repository and the script does not need to be run unless additional background samples are to be added to the 4 tau studies.
To obtain the json files simply run the following command within `Studies_4tau/UnifiedProcessor_Dir` while on the Wisconsin AF:
`python3 PreSkimWeight_Runner.py`.
The script will produce json files `genWeightSum_2018_WithQCD_JSON.json` and `numEvents_2018_WithQCD_JSON.json` which contain the sum of gen weights for each background process prior to skimming and the number of events from each background process prior to skimming respectively.
