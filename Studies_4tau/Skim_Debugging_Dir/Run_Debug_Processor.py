from Debug_Cut_Processor import PlottingScriptProcessor as Processor

import awkward as ak
import uproot
import hist
from hist import intervals
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from coffea import processor, nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import candidate, vector
from coffea import util
from math import pi
import numba 
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import vector
import os
import time
import datetime
from distributed import Client
from dask_jobqueue import HTCondorCluster
import csv
import argparse

#X509 function (for HTC)
def move_X509():
	try:
		_x509_localpath = (
			[
				line
				for line in os.popen("voms-proxy-info").read().split("\n")
				if line.startswith("path")
			][0]
			.split(":")[-1]
			.strip()
		)
	except Exception as err:
		raise RuntimeError(
			"x509 proxy could not be parsed, try creating it with 'voms-proxy-init'"
		) from err
	_x509_path = f'/scratch/{os.environ["USER"]}/{_x509_localpath.split("/")[-1]}'
	os.system(f"cp {_x509_localpath} {_x509_path}")
	return os.path.basename(_x509_localpath)


#Parse command line arguements
parse = argparse.ArgumentParser()
parse.add_argument("-f", "--File", help = "Input File (1 = mine, 0 = Ganesh)")
parse.add_argument("-d", "--Debug", help = "Debug Code integer between 0 and 8")
args= parse.parse_args()

if __name__ == "__main__":
	#Condor related stuff
	os.environ["CONDOR_CONFIG"] = "/etc/condor/condor_config"
	
	#Xrootd crap
	_x509_path = move_X509()
	print(f"x509 path: {_x509_path}")
	htc_log_err_dir = "/scratch/twnelson/ControlPlot_HTC/Run_" + str(time.localtime()[0]) + "_" + str(time.localtime()[1]) + "_" + str(time.localtime()[2]) + "_" + str(time.localtime()[3]) + f".{time.localtime()[4]:02d}"
	os.makedirs(htc_log_err_dir)

	cluster = HTCondorCluster(
			cores=1,
			memory="4 GB",
			disk="2.0 GB",
			death_timeout = '60',
			job_extra_directives={
				"+JobFlavour": '"tomorrow"',
				"log": "dask_job_output.$(PROCESS).$(CLUSTER).log",
				"output": "dask_job_output.$(PROCESS).$(CLUSTER).out",
				"error": "dask_job_output.$(PROCESS).$(CLUSTER).err",
				"should_transfer_files": "yes",
				"when_to_transfer_ouput": "ON_EXIT_OR_EVICT",
				"transfer_executable": "false",
				"+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest-py3.10"',
				"Requirements": "HasSingularityJobStart",
				"InitialDir": f'/scratch/{os.environ["USER"]}',
				'transfer_input_files': f"{_x509_path}",

			},
			job_script_prologue = [
				"export XRD_RUNFORKHANDLER=1",
				f"export X509_USER_PROXY={_x509_path}",
			]
	)
	cluster.adapt(minimum=1, maximum=500)

	run_on_condor = True 
	
	if (run_on_condor):
		print("Run on Condor")
		runner = processor.Runner(
			executor = processor.DaskExecutor(client=Client(cluster),status=False),
			schema=BaseSchema,
			skipbadfiles=True,
			xrootdtimeout=1000,
            #chunksize=500000,
            #maxchunks = 1
		)
	else: #Iterative runner
		runner = processor.Runner(executor = processor.IterativeExecutor(), schema=BaseSchema)
	

	debug_num = int(args.Debug)
	file_code= int(args.File)

	#Get number of taus to be selected
	if (debug_num <= 4):
		n_tau = 0
	if (debug_num == 5):
		n_tau = 1
	if (debug_num == 6):
		n_tau = 2
	if (debug_num == 7):
		n_tau = 3
	if (debug_num == 8):
		n_tau = 4

	#Diretory for files
	Skimmed_Ganesh_base = "root://cmsxrootd.hep.wisc.edu//store/user/gparida/HHbbtt/Hadded_Skimmed_Files/Full_Production_CMSSW_13_0_13_Nov24_23/LooseSelection_MET_gt_80_nFatJet_gt_0_Skim/2018/"	
	Skimmed_4tau_base_Data = "root://cmsxrootd.hep.wisc.edu//store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/Data/"

	#Select file dict based upon file_code passed in via command line
	#file_dict_data_4tau = {
	#	"Data_Mu": [Skimmed_4tau_base_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim/SingleMu_Run2018A.root"]
	#}
	file_dict_data_4tau = {
		"Data_Mu": [Skimmed_4tau_base_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root"]
			#	Skimmed_4tau_base_Data + "SingleMu_Run2018B_15January26_0731_skim_Jan26Skim/SingleMu_Run2018B.root",
			#	Skimmed_4tau_base_Data + "SingleMu_Run2018C_15January26_0740_skim_Jan26Skim/SingleMu_Run2018C.root",
			#	Skimmed_4tau_base_Data + "SingleMu_Run2018D_15January26_0815_skim_Jan26Skim/SingleMu_Run2018D.root"]
	}
	file_dict_data_2b2tau = {
		"Data_Mu": [Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018A.root",
				#	Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018B.root",
				#	Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018C.root",
				#	Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018D.root",
					]
	}

	if (file_code == 0):
		file_dict = file_dict_data_2b2tau 
	else:
		file_dict = file_dict_data_4tau 

	#Logic to save the coffea fill
	tau_selec = f"{n_tau}_BoostedTaus"

	if (file_code == 0):
		file_source_str = "2b2Tau_Samples"
	else:
		file_source_str = "4Tau_Samples"

	debug_str_dict = {
			0: "No_Selections",
			1: "MET_Selections",
			2: "FatJet_Selections",
			3: "Flag_Selections",
			4: "PV_Selections",
			5: "All2b2Tau_Selections",
			6: "All2b2Tau_Selections",
			7: "All2b2Tau_Selections",
			8: "All2b2Tau_Selections"
		}

	file_name = "output_" + file_source_str + "_" + debug_str_dict[debug_num] + "_" + tau_selec + "_FullSingeMu2018Data.coffea"
	
	#Run the processor
	start_time = time.time()
	debug_out = runner(file_dict, treename="Events", processor_instance=Processor(nBoostedTaus = n_tau, ApplyTrigger = False, DebugCode = debug_num))
	end_time = time.time()

	run_time = end_time - start_time

	print("It took %d s to run"%run_time)

	outfile = os.path.join(os.getcwd(), file_name)
	util.save(debug_out, outfile)
	print(f"Saved output to {outfile}")




