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
import json
from Processors import Skim_Table_Processor as SkimProcessor
import cowtools.jobqueue
import cloudpickle

#Dictionary of cross sections 
xSection_Dictionary = {"Signal": 0.01, #Chosen to make plots readable
						"TTTo2L2Nu": 87.5595, "TTToSemiLeptonic": 365.2482, "TTToHadronic": 381.0923,
						
						#DiBoson Background
						"ZZ2l2q": 3.676, "WZ2l2q": 6.565, "WZ1l1nu2q": 9.119, "WZ1l3nu": 3.414, "VV2l2nu": 11.09, "WWTo1L1Nu2Q": 51.65, "WWTo4Q": 51.03, "ZZTo4Q": 3.262, "ZZTo2L2Nu": 0.9738, "ZZTo2Nu2Q": 4.545, #"WZ3l1nu.root" : 27.57,
						
						#ZZ->4l
						"ZZ4l": 1.325,
						
						"Tbar-tchan": 80.8, "T-tchan": 134.2, "Tbar-tW": 39.65, "T-tW": 39.65, "ST_s-channel_4f_leptonDecays": 3.588, "ST_s-channel_4f_hadronicDecays": 7.485,
						#Drell-Yan Jets
						"DYJetsToLL_M-4to50_HT-70to100": 314.8,
						"DYJetsToLL_M-4to50_HT-100to200": 190.6,
						"DYJetsToLL_M-4to50_HT-200to400": 42.27,
						"DYJetsToLL_M-4to50_HT-400to600": 4.05,
						"DYJetsToLL_M-4to50_HT-600toInf": 1.216,
						"DYJetsToLL_M-50_HT-70to100": 140.0,
						"DYJetsToLL_M-50_HT-100to200": 139.2,
						"DYJetsToLL_M-50_HT-200to400": 38.4,
						"DYJetsToLL_M-50_HT-400to600": 5.174,
						"DYJetsToLL_M-50_HT-600to800": 1.258,
						"DYJetsToLL_M-50_HT-800to1200": 0.5598,
						"DYJetsToLL_M-50_HT-1200to2500": 0.1305,
						"DYJetsToLL_M-50_HT-2500toInf": 0.002997,
						
						#WJets
						"WJetsToLNu_HT-70To100":1283.0, "WJetsToLNu_HT-100To200" : 1244.0, "WJetsToLNu_HT-200To400": 337.8, "WJetsToLNu_HT-400To600": 44.93, "WJetsToLNu_HT-600To800": 11.19, "WJetsToLNu_HT-800To1200": 4.926, "WJetsToLNu_HT-1200To2500" : 1.152, "WJetsToLNu_HT-2500ToInf" : 0.02646, 
						#SM Higgs
						"ZH125": 0.7544*0.0621, "ggZHLL125":0.1223 * 0.062 * 3 * 0.033658, "ggZHNuNu125": 0.1223*0.062*0.2,"ggZHQQ125": 0.1223*0.062*0.6991, "toptopH125": 0.5033*0.062, #"ggH125": 48.30* 0.0621, "qqH125": 3.770 * 0.0621, "WPlusH125": 
						
						#QCD
						"QCD_HT50to100": 186100000.0, "QCD_HT100to200": 23630000.0, "QCD_HT200to300": 1554000.0, "QCD_HT300to500": 323800.0, "QCD_HT500to700": 30280.0, 
						"QCD_HT700to1000": 6392.0, "QCD_HT1000to1500": 1118.0, "QCD_HT1500to2000": 108.9, "QCD_HT2000toInf": 21.93,   
						}	

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

row_name_var_dict = {"Events after selection": "Event_Count", "Gen Weight After Selection": "SumGenWeight", "Yield": "Yield"}

#Function to control logic for filling the table
def table_maker(sample_name, row_name, coffea_object, genWeight_Dict, Event_Dict):
	table_entry = -1
	if (row_name == "Process"):
		table_entry = sample_name
	elif (row_name == "CrossSec"):
		if ("Data" in sample_name):
			table_entry = "N/A"
		else:
			table_entry = "%.2f"%xSection_Dictionary[sample_name]
	elif (row_name == "Events Before Skim"):
		if ("Data" in sample_name):
			table_entry = "Not Stored"
		else:
			table_entry = "%d"%Event_Dict[sample_name]
	elif (row_name == "Gen Weight Before Skim"):
		if ("Data" in sample_name):
			table_entry = "Not Stored"
		else:
			table_entry = "%d"%genWeight_Dict[sample_name]
	#if (row_name == "Events after selection" or row_name == "Gen Weight After Selection" or row_name == "Yield"):
	else:
		table_entry = "%.2f"%coffea_object[sample_name][row_name_var_dict[row_name]]
	
	return table_entry

if __name__ == "__main__":
	#Condor related stuff
	run_on_condor = True
	os.environ["CONDOR_CONFIG"] = "/etc/condor/condor_config"
	
	#Xrootd setup
	_x509_path = move_X509()
	print(f"x509 path: {_x509_path}")
	htc_log_err_dir = "/scratch/twnelson/ControlPlot_HTC/Run_" + str(time.localtime()[0]) + "_" + str(time.localtime()[1]) + "_" + str(time.localtime()[2]) + "_" + str(time.localtime()[3]) + f".{time.localtime()[4]:02d}"
	os.makedirs(htc_log_err_dir)

	cluster = HTCondorCluster(
			cores=1,
			memory="4 GB",
			disk="2 GB",
			death_timeout = '60',
            #python = "/usr/local/bin/python3",
			job_extra_directives={
				"+JobFlavour": '"tomorrow"',
				"log": "dask_job_output.$(PROCESS).$(CLUSTER).log",
				"output": "dask_job_output.$(PROCESS).$(CLUSTER).out",
				"error": "dask_job_output.$(PROCESS).$(CLUSTER).err",
				"should_transfer_files": "yes",
				"when_to_transfer_ouput": "ON_EXIT_OR_EVICT",
				"transfer_executable": "false",
			#	"+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux9:0.7.25-py3.10"',
			#	"Requirements": "HasSingularityJobStart",
				
				#"+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest-py3.10"',
				#"+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux9:0.7.25-py3.10"',
				#"Requirements": "HasSingularityJobStart",
				#"container_image": "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest-py3.10",
				"container_image": "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-base-almalinux9:0.7.25-py3.10",
				"InitialDir": f'/scratch/{os.environ["USER"]}',
				'transfer_input_files': f"{_x509_path}",

			},
			job_script_prologue = [
				"export XRD_RUNFORKHANDLER=1",
				f"export X509_USER_PROXY={_x509_path}",
			]
	)
	cluster.adapt(minimum=1, maximum=500)

#	cluster = cowtools.jobqueue.GetCondorClient(
#					memory = "4 GB",
#					disk = "2 GB",
#					max_workers=500,
#					container_image = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest-py3.10"
#				)
	
	if (run_on_condor):
		print("Run on Condor")
		runner = processor.Runner(
			executor = processor.DaskExecutor(client=Client(cluster),status=False),
			#executor = processor.DaskExecutor(client=cluster,status=False),
			schema=BaseSchema,
			skipbadfiles=True,
			xrootdtimeout=1000,
            #chunksize=500000,
            #maxchunks = 1
		)

		#Pass modules to HTC
		cloudpickle.register_pickle_by_value(SkimProcessor)
    
	else: #Iterative runner
		print("Run Iteratively")
		runner = processor.Runner(executor = processor.IterativeExecutor(), schema=BaseSchema)

	#Diretory for files
	Skimmed_4tau_base_MC = "root://cmsxrootd.hep.wisc.edu//store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/"
	Skimmed_4tau_base_Data = "root://cmsxrootd.hep.wisc.edu//store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/Data/"
	Skimmed_4tau_loc_Data = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/Data/"
	Skimmed_4tau_loc_MC = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/"

	#Make full arrays of single Muon data
	SingleMu_2018A = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root") 
	SingleMu_2018B = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018B_15January26_0731_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root") 
	SingleMu_2018C = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018C_15January26_0740_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root") 
	SingleMu_2018D = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018D_15January26_0815_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root") 
	
	MET_2018A = glob.glob(Skimmed_4tau_loc_Data + "MET_Run2018A_14April26_1337_skim_4Tau_Selections/singleFileSkimForSubmission-NANO_NANO_*.root") 
	MET_2018B = glob.glob(Skimmed_4tau_loc_Data + "MET_Run2018B_14April26_1334_skim_4Tau_Selections/singleFileSkimForSubmission-NANO_NANO_*.root") 
	MET_2018C = glob.glob(Skimmed_4tau_loc_Data + "MET_Run2018C_14April26_1331_skim_4Tau_Selections/singleFileSkimForSubmission-NANO_NANO_*.root") 
	MET_2018D = glob.glob(Skimmed_4tau_loc_Data + "MET_Run2018D_14April26_1343_skim_4Tau_Selections/singleFileSkimForSubmission-NANO_NANO_*.root") 

	#Single MuonA debugging production
	SingleMu_2018A_Debug = glob.glob("/hdfs/store/user/twnelson/HH4Tau_EtAl/SkimDebugging/SingleMu_Run2018A_24March26_0456_skim_4TauFixed_NonEmpty/singleFileSkimForSubmission-NANO_NANO_*.root")


	#Offline debugging to test code for bugs
	SingleMu_2018A_Offline_SingleFile = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_12.root")
	ZZ4L_2018_Offline_SingleFile = glob.glob(Skimmed_4tau_loc_MC + "ZZTo4L_26August25_0757_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_12.root")

	#Make full arrays of backgrounds
	TTToSemiLeptonic_2018 = glob.glob(Skimmed_4tau_loc_MC + "TTToSemiLeptonic_35August25_0448_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	TTTo2L2Nu_2018 = glob.glob(Skimmed_4tau_loc_MC + "TTTo2L2Nu_26August25_0719_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	TTToHadronic_2018 = glob.glob(Skimmed_4tau_loc_MC + "TTToHadronic_25October25_0813_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZ4L_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo4L_26August25_0757_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZTo2L2Nu_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo2L2Nu_04March26_0503_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZTo2L2Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo2Q2L_26August25_1034_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZTo2Nu2Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo2Nu2Q_04March26_0510_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZTo4Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo4Q_04March26_0505_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	VV2l2nu_2018 = glob.glob(Skimmed_4tau_loc_MC + "WWTo2L2Nu_26August25_1040_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WWTo1L1Nu2Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "WWTo2L2Nu_26August25_1040_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WWTo4Q_2018 = glob.glob(Skimmed_4tau_loc_MC + "WWTo4Q_04March26_0512_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WZ1l3nu_2018 = glob.glob(Skimmed_4tau_loc_MC + "WZTo1L3Nu_4f_26August25_1016_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ZZ2l2q_2018 = glob.glob(Skimmed_4tau_loc_MC + "ZZTo2Q2L_26August25_1034_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WZ2l2q_2018 = glob.glob(Skimmed_4tau_loc_MC + "WZTo2L2Q_26August25_0926_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WZ1l1nu2q_2018 = glob.glob(Skimmed_4tau_loc_MC + "WZTo1L1Nu2Q_26August25_0840_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT70to100_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-70to100_12December25_1606_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT100to200_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-100to200_12December25_1604_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT200to400_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-200to400_12December25_1544_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT400to600_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-400to600_12December25_1552_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M4to50_HT600toInf_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-4to50_HT-600toInf_12December25_1608_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT70to100_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-70to100_12December25_1556_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT100to200_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-100to200_12December25_1548_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT200to400_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-200to400_12December25_1559_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT400to600_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-400to600_12December25_1546_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT600to800_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-600to800_12December25_1555_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT800to1200_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-800to1200_12December25_1602_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT1200to2500_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-1200to2500_12December25_1547_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	DYJetsToLL_M50_HT2500toInf_2018 = glob.glob(Skimmed_4tau_loc_MC + "DYJetsToLL_M-50_HT-1200to2500_12December25_1547_skim_Oldskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	Ttchan_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_t-channel_top_4f_InclusiveDecays_26August25_0843_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	Tbartchan_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_t-channel_antitop_4f_InclusiveDecays_26August25_0821_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	TtW_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_tW_top_5f_inclusiveDecays_26August25_0753_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	TbartW_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_tW_antitop_5f_inclusiveDecays_26August25_1030_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ST_schannel_4f_hadronicDecays_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_s-channel_4f_hadronicDecays_04March26_0506_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	ST_schannel_4f_leptonDecays_2018 = glob.glob(Skimmed_4tau_loc_MC + "ST_s-channel_4f_leptonDecays_04March26_0507_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WJetsToLNu_HT70To100_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-70To100_04March26_0515_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WJetsToLNu_HT100To200_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-100To200_26August25_0810_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WJetsToLNu_HT200To400_2018 = glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-200To400_26August25_0709_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root")
	WJetsToLNu_HT400To600_2018 = np.append(glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-400To600_26August25_1014_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-400To600_OtherPart_26August25_1032_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))
	WJetsToLNu_HT600To800_2018 = np.append(glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-600To800_26August25_0755_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-600To800_OtherPart_26August25_0752_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))
	WJetsToLNu_HT800To1200_2018 = np.append(glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-800To1200_26August25_0708_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-800To1200_OtherPart_26August25_0925_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))
	WJetsToLNu_HT1200To2500_2018 = np.append(glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-120)0To2500_26August25_1016_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-1200To2500_OtherPart_26August25_1041_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))
	WJetsToLNu_HT2500ToInf_2018 = np.append(glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-2500ToInf_26August25_1047_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"),
		glob.glob(Skimmed_4tau_loc_MC + "WJetsToLNu_HT-2500ToInf_OtherPart_26August25_1043_skim_Newskim/singleFileSkimForSubmission-NANO_NANO_*.root"))
	QCD_HT50To100 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT50to100_23April26_0525_skim_FourTauSkim/singleFileSkimForSubmission-NANO_NANO_*.root")
	QCD_HT100To200 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT100to200_23April26_0519_skim_FourTauSkim/singleFileSkimForSubmission-NANO_NANO_*.root")
	QCD_HT200To300 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT200to300_23April26_0542_skim_FourTauSkim/singleFileSkimForSubmission-NANO_NANO_*.root")
	QCD_HT300To500 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT300to500_23April26_0555_skim_FourTauSkim/singleFileSkimForSubmission-NANO_NANO_*.root")
	QCD_HT500To700 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT500to700_23April26_0512_skim_FourTauSkim/singleFileSkimForSubmission-NANO_NANO_*.root")
	QCD_HT700To1000 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT700to1000_23April26_0528_skim_FourTauSkim/singleFileSkimForSubmission-NANO_NANO_*.root")
	QCD_HT1000To1500 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT1000to1500_23April26_0536_skim_FourTauSkim/singleFileSkimForSubmission-NANO_NANO_*.root")
	QCD_HT1500To2000 = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT1500to2000_23April26_0539_skim_FourTauSkim/singleFileSkimForSubmission-NANO_NANO_*.root")
	QCD_HT2000ToInf = glob.glob(Skimmed_4tau_loc_MC + "QCD_HT2000toInf_23April26_0541_skim_FourTauSkim/singleFileSkimForSubmission-NANO_NANO_*.root")

	file_dict_data_test = {
		#"Data_Mu" : [Skimmed_4tau_base_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim/SingleMu_Run2018A.root"]
		#"Data_Mu": [Skimmed_4tau_base_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root"]
		#"Data_Mu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in SingleMu_2018A_Debug],
		#"Data_Mu": [file for file in SingleMu_2018A_Debug] 
		"Data_Mu": ["root://cmsxrootd.hep.wisc.edu//store/user/twnelson/HH4Tau_EtAl/SkimDebugging/SingleMu_Run2018A_24March26_0937_skim_NullSkimming/singleFileSkimForSubmission-NANO_NANO_402.root"] #Run a single file offline
	}

	file_dict_full = {
			"TTToSemiLeptonic": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in TTToSemiLeptonic_2018],
			"TTTo2L2Nu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in TTTo2L2Nu_2018],
			"TTToHadronic": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in TTToHadronic_2018],
			"ZZ4l": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZ4L_2018],
			"ZZTo2L2Nu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZTo2L2Nu_2018],
			"ZZTo2Nu2Q": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZTo2Nu2Q_2018],
			"VV2l2nu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in VV2l2nu_2018],
			"ZZTo4Q" : ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZTo4Q_2018],
			"WWTo1L1Nu2Q": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WWTo1L1Nu2Q_2018],
			"WWTo4Q": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WWTo4Q_2018],
			"WZ1l3nu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WZ1l3nu_2018],
			"ZZ2l2q": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ZZ2l2q_2018],
			"WZ2l2q": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WZ2l2q_2018],
			"WZ1l1nu2q" : ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WZ1l1nu2q_2018],
			"DYJetsToLL_M-4to50_HT-70to100": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M4to50_HT70to100_2018],
			"DYJetsToLL_M-4to50_HT-100to200": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M4to50_HT100to200_2018],
			"DYJetsToLL_M-4to50_HT-200to400": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M4to50_HT200to400_2018],
			"DYJetsToLL_M-4to50_HT-400to600": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M4to50_HT400to600_2018],
			"DYJetsToLL_M-4to50_HT-600toInf":["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M4to50_HT600toInf_2018],
			"DYJetsToLL_M-50_HT-70to100": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT70to100_2018],
			"DYJetsToLL_M-50_HT-100to200": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT100to200_2018],
			"DYJetsToLL_M-50_HT-200to400": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT200to400_2018],
			"DYJetsToLL_M-50_HT-400to600": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT400to600_2018],
			"DYJetsToLL_M-50_HT-600to800": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT600to800_2018],
			"DYJetsToLL_M-50_HT-800to1200": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT800to1200_2018],
			"DYJetsToLL_M-50_HT-1200to2500": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT1200to2500_2018],
			"DYJetsToLL_M-50_HT-2500toInf": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in DYJetsToLL_M50_HT2500toInf_2018],
			"T-tchan": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in Ttchan_2018],
			"Tbar-tchan": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in Tbartchan_2018],
			"T-tW": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in TtW_2018],
			"Tbar-tW": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in TbartW_2018],
			"ST_s-channel_4f_hadronicDecays": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ST_schannel_4f_hadronicDecays_2018],
			"ST_s-channel_4f_leptonDecays": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in ST_schannel_4f_leptonDecays_2018],
			"WJetsToLNu_HT-70To100": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT70To100_2018],
			"WJetsToLNu_HT-100To200": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT100To200_2018],
			"WJetsToLNu_HT-200To400": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT200To400_2018],
			"WJetsToLNu_HT-400To600": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT400To600_2018],
			"WJetsToLNu_HT-600To800": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT600To800_2018],
			"WJetsToLNu_HT-800To1200": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT800To1200_2018],
			"WJetsToLNu_HT-1200To2500": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT1200To2500_2018],
			"WJetsToLNu_HT-2500ToInf": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in WJetsToLNu_HT2500ToInf_2018],
			"QCD_HT50to100": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in QCD_HT50To100],
			"QCD_HT100to200": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in QCD_HT100To200],
			"QCD_HT200to300": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in QCD_HT200To300],
			"QCD_HT300to500": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in QCD_HT300To500],
			"QCD_HT500to700": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in QCD_HT500To700],
			"QCD_HT700to1000": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in QCD_HT700To1000],
			"QCD_HT1000to1500": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in QCD_HT1000To1500],
			"QCD_HT1500to2000": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in QCD_HT1500To2000],
			"QCD_HT2000toInf": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in QCD_HT2000ToInf],
			#"Data_Mu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in np.append(SingleMu_2018A, np.append(SingleMu_2018B, np.append(SingleMu_2018C,SingleMu_2018D)))]
			"Data_MuA": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in SingleMu_2018A],
			"Data_MuB": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in SingleMu_2018B],
			"Data_MuC": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in SingleMu_2018C],
			"Data_MuD": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in SingleMu_2018D]
			#"Data_MET": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in np.append(MET_2018A, np.append(MET_2018B, np.append(MET_2018C,MET_2018D)))]
		}
	
	#Set file dictionary and list of backgrounds prior to running processor
	#file_dict = file_dict_data_test
	file_dict = file_dict_full

	#Pull in the weight and event count prior to skimming information
	#with open("genWeightSum_JSON.json") as json_file:
	with open("genWeightSum_2018_WithQCD_WithData_JSON.json") as json_file:
		sumWEvents_Dict = json.load(json_file)

	with open("numEvents_2018_WithQCD_WithData_JSON.json") as json_file:
		numEvents_Dict = json.load(json_file)
	

	start_time = time.time()
	
	n_taus = 4
	print("About to run processor")
	start_time = time.time()
	fourtau_out = runner(file_dict, treename="Events", processor_instance=SkimProcessor.TableProcessor(sumWEvents_Dict = sumWEvents_Dict, nBoostedTaus = n_taus, ApplyTrigger = True)) #Modified for NanoAOD (changd treename)
	end_time = time.time()
	
	time_running = end_time-start_time
	print("It takes about %.1f s to run the coffea processor with %d boosted tau selections"%(time_running,n_taus))

	#Set up dictionary to create csv file
	row_names = ["Process","CrossSec","Events Before Skim","Gen Weight Before Skim","Events after selection", "Gen Weight After Selection", "Yield"]
	sample_names = list(file_dict.keys())

	csv_array = []

	for sample in sample_names:
		csv_dict = dict.fromkeys(row_names)
		for row in row_names:
			csv_dict[row] = table_maker(sample, row, fourtau_out, sumWEvents_Dict, numEvents_Dict)
		csv_array.append(csv_dict)

    #Save table
	with open ("DebuggingTable.csv", "w", newline="") as out_file:
		dict_write = csv.DictWriter(out_file, row_names)
		dict_write.writeheader()
		dict_write.writerows(csv_array)
	
    #Save coffea file
	#outfile = os.path.join(os.getcwd() + "/Output_4Tau/", f"output_{n_taus}_boosted_tau_selec_SingleMuData_4TauSamples_WithSingleMuTrigger.coffea")
#	outfile = os.path.join(os.getcwd() + "/Output_4Tau/", f"output_{n_taus}_boosted_tau_selec_SingleMuData_4TauSamples_WithSingleMuTrigger_WithQCD.coffea")
#	util.save(fourtau_out, outfile)
#	print(f"Saved output to {outfile}")	
