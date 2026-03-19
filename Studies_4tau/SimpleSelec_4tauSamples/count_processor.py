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
import glob

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

class CountingProcessor(processor.ProcessorABC):
	def __init__(self):
		pass
	def process(self,events):
		dataset = events.metadata['dataset']
		event_count = ak.num(events.boostedTau_pt,axis=0)

		return {dataset: {"n_events": event_count}}

	def postprocess(self, accumulator):
		pass

if __name__ == "__main__":
	print("Main")
	
	Skimmed_4tau_loc_Data = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/Data/"
	Ganesh_Skim_Loc_Data = "/hdfs/store/user/twnelson/HH4Tau_EtAl/SkimDebugging/SingleMu_Run2018A_17March26_0712_skim_Ganesh_Selections/"

	#Make full array of input files
	#input_files = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root") 
	input_files = glob.glob(Ganesh_Skim_Loc_Data + "singleFileSkimForSubmission-NANO_NANO_*.root")

	file_dict = {
		"Data_Mu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in input_files] 
		#"Data_Mu": input_files 
	}
	
	#Condor related stuff
	os.environ["CONDOR_CONFIG"] = "/etc/condor/condor_config"
	
	#Xrootd crap
	_x509_path = move_X509()
	print(f"x509 path: {_x509_path}")
	htc_log_err_dir = "/scratch/twnelson/ControlPlot_HTC/Run_" + str(time.localtime()[0]) + "_" + str(time.localtime()[1]) + "_" + str(time.localtime()[2]) + "_" + str(time.localtime()[3]) + f".{time.localtime()[4]:02d}"
	os.makedirs(htc_log_err_dir)

	cluster = HTCondorCluster(
			cores=1,
			memory="8 GB",
			disk="4 GB",
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

	print(f"https://cms01.hep.wisc.edu:8004/user/{os.environ['USER']}/{cluster.dashboard_link}")

	start_time = time.time()
	fourtau_out = runner(file_dict, treename="Events", processor_instance=CountingProcessor()) 
	end_time = time.time()
	
	time_running = end_time-start_time
	print("It takes about %.1f s to run the coffea processor with %d boosted tau selections"%(time_running,4))
	print("There are " + str(fourtau_out["Data_Mu"]["n_events"]) + " events")

