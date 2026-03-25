import awkward as ak
import uproot
import hist
from hist import intervals
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from coffea import processor, nanoevents
from coffea.analysis_tools import PackedSelection
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
from Processors import Skim_Emulation_CoffeaProcessor as SkimProcessor

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

if __name__ == "__main__":
	#Condor related stuff
	os.environ["CONDOR_CONFIG"] = "/etc/condor/condor_config"
	
	#Xrootd
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
                "container_image": "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest-py3.10",
				#"+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest-py3.10"',
				#"Requirements": "HasSingularityJobStart",
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
	
	four_tau_hist_list = [
			"boostedtau_pt_Trigg","Leadingboostedtau_pt_Trigg", "Subleadingboostedtau_pt_Trigg", "Thirdleadingboostedtau_pt_Trigg","Fourthleadingboostedtau_pt_Trigg",
			"boostedtau_eta_Trigg","boostedtau_phi_Trigg",
			#"boostedtau_pt_Trigg","Leadingboostedtau_pt_Trigg","boostedtau_eta_Trigg","boostedtau_phi_Trigg",
			"tau_pt_Trigg","Leadingtau_pt_Trigg","tau_eta_Trigg","tau_phi_Trigg",
			"electron_pt_Trigg","Leadingelectron_pt_Trigg","electron_eta_Trigg","electron_phi_Trigg",
			"muon_pt_Trigg","Leadingmuon_pt_Trigg","muon_eta_Trigg","muon_phi_Trigg",
			"Jet_pt_Trigg","LeadingJet_pt_Trigg","Jet_eta_Trigg","Jet_phi_Trigg",
			"AK8Jet_pt_Trigg","LeadingAK8Jet_pt_Trigg","AK8Jet_eta_Trigg","AK8Jet_phi_Trigg",
			"MET","HT","MHT",
			] 

	hist_name_dict = {
					"boostedtau_pt_Trigg": r"Boosted $\tau$ $p_T$ after Trigger"," Leadingboostedtau_pt_Trigg": r"Leading boosted $\tau$ $p_T$ after Trigger",
					"Subleadingboostedtau_pt_Trigg": r"Subleading boosted $\tau$ $p_T$ after Trigger",
					"Thirdleadingboostedtau_pt_Trigg": r"3rd leading boosted $\tau$ $p_T$ after Trigger",
					"Fourthleadingboostedtau_pt_Trigg": r"4th leading boosted $\tau$ $p_T$ after Trigger",
					"boostedtau_eta_Trigg": r"Boosted $\tau$ $\eta$ after Trigger","boostedtau_phi_Trigg": r"Boosted $\tau$ $\phi$ after Trigger", 
					"tau_pt_Trigg": r"$\tau$ $p_T$ after Trigger","Leadingtau_pt_Trigg": r"Leading $\tau$ $p_T$ after Trigger",
					"tau_eta_Trigg": r"$\tau$ $\eta$ after Trigger","tau_phi_Trigg": r"$\tau$ $\phi$ after Trigger", 
					"electron_pt_Trigg": r"e $p_T$ after Trigger","Leadingelectron_pt_Trigg": r"Leading e $p_T$ after Trigger",
					"electron_eta_Trigg": r"e $\eta$ after Trigger", "electron_phi_Trigg": r"e $\phi$ after Trigger", 
					"muon_pt_Trigg": r"$\mu$ $p_T$ after Trigger","Leadingmuon_pt_Trigg": r"Leading $\mu$ $p_T$ after Trigger",
					"muon_eta_Trigg": r"$\mu$ $\eta$ after Trigger","muon_phi_Trigg": r"$\mu$ $\phi$ after Trigger", 
					"Jet_pt_Trigg": r"Jet $p_T$ after Trigger","LeadingJet_pt_Trigg": r"LeadingJet $p_T$ after Trigger",
					"Jet_eta_Trigg": r"Jet $\eta$ after Trigger", "Jet_phi_Trigg": r"Jet $\phi$ after Trigger", 
					"AK8Jet_pt_Trigg": r"AK8Jet $p_T$ after Trigger","LeadingAK8Jet_pt_Trigg": r"Leading AK8Jet $p_T$ after Trigger",
					"AK8Jet_eta_Trigg": r"AK8Jet $\eta$ after Trigger","AK8Jet_phi_Trigg": r"AK8Jet $\phi$ after Trigger",
					"MET": r"MET after Trigger", "HT": r"HT after Trigger", "MHT": r"MHT after Trigger",
					}

	#Diretory for files
	Skimmed_Ganesh_base = "root://cmsxrootd.hep.wisc.edu//store/user/gparida/HHbbtt/Hadded_Skimmed_Files/Full_Production_CMSSW_13_0_13_Nov24_23/LooseSelection_MET_gt_80_nFatJet_gt_0_Skim/2018/"
	
	file_dict_test = {
			"ZZ4l": [Skimmed_Ganesh_base + "ZZTo4L.root"],
			"Data_Mu": [Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018A.root"]
		}

	My_2b2Tau_Selections = glob.glob("/hdfs/store/user/twnelson/HH4Tau_EtAl/SkimDebugging/SingleMu_Run2018A_17March26_0712_skim_Ganesh_Selections/singleFileSkimForSubmission-NANO_NANO_*.root")
	#My_2b2Tau_Selections = glob.glob("/hdfs/store/user/twnelson/HH4Tau_EtAl/SkimDebugging/SingleMu_Run2018A_17March26_0934_skim_Ganesh_Selections_UpdatedSkimmer1/singleFileSkimForSubmission-NANO_NANO_*.root ")
	
	file_dict_data_test = {
			#"Data_Mu": [Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018A.root"]
			#"Data_Mu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in My_2b2Tau_Selections]
			"Data_Mu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in My_2b2Tau_Selections]
		}
	
	file_dict_full = {
			"TTToSemiLeptonic": list(np.append([Skimmed_Ganesh_base + "TTToSemiLeptonic_" + str(j) + ".root" for j in range(2,5)], Skimmed_Ganesh_base + "TTToSemiLeptonic.root")),
			"TTTo2L2Nu": [Skimmed_Ganesh_base + "TTTo2L2Nu.root", Skimmed_Ganesh_base + "TTTo2L2Nu_2.root"],
			"TTToHadronic": [Skimmed_Ganesh_base + "TTToHadronic.root"],
			"ZZ4l": [Skimmed_Ganesh_base + "ZZTo4L.root"],
			"ZZTo2L2Nu": [Skimmed_Ganesh_base + "ZZTo2L2Nu.root"],
			"ZZTo2Nu2Q": [Skimmed_Ganesh_base + "ZZTo2Nu2Q_5f.root"],
			"ZZTo4Q": [Skimmed_Ganesh_base + "ZZTo4Q_5f.root"],
			"VV2l2nu": [Skimmed_Ganesh_base + "WWTo2L2Nu.root"],
			"WWTo1L1Nu2Q": [Skimmed_Ganesh_base + "WWTo1L1Nu2Q_4f.root"],
			"WWTo4Q": [Skimmed_Ganesh_base + "WWTo4Q_4f.root"],
			"WZ1l3nu": [Skimmed_Ganesh_base + "WZTo1L3Nu_4f.root"],
			"ZZ2l2q": [Skimmed_Ganesh_base + "ZZTo2Q2L_mllmin4p0.root"],
			"WZ2l2q": [Skimmed_Ganesh_base + "WZTo2Q2L_mllmin4p0.root"],
			"WZ1l1nu2q" : [Skimmed_Ganesh_base + "WZTo1L1Nu2Q_4f.root"],
			"DYJetsToLL_M-4to50_HT-70to100": [Skimmed_Ganesh_base + "DYJetsToLL_M-4to50_HT-70to100.root"],
			"DYJetsToLL_M-4to50_HT-100to200": [Skimmed_Ganesh_base + "DYJetsToLL_M-4to50_HT-100to200.root"],
			"DYJetsToLL_M-4to50_HT-200to400": [Skimmed_Ganesh_base + "DYJetsToLL_M-4to50_HT-200to400.root"],
			"DYJetsToLL_M-4to50_HT-400to600": [Skimmed_Ganesh_base + "DYJetsToLL_M-4to50_HT-400to600.root"],
			"DYJetsToLL_M-4to50_HT-600toInf": [Skimmed_Ganesh_base + "DYJetsToLL_M-4to50_HT-600toInf.root"],
			"DYJetsToLL_M-50_HT-70to100": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-70to100.root"],
			"DYJetsToLL_M-50_HT-100to200": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-100to200.root"],
			"DYJetsToLL_M-50_HT-200to400": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-200to400.root"],
			"DYJetsToLL_M-50_HT-400to600": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-400to600.root"],
			"DYJetsToLL_M-50_HT-600to800": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-600to800.root"],
			"DYJetsToLL_M-50_HT-800to1200": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-800to1200.root"],
			"DYJetsToLL_M-50_HT-1200to2500": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-1200to2500.root"],
			"DYJetsToLL_M-50_HT-2500toInf": [Skimmed_Ganesh_base + "DYJetsToLL_M-50_HT-2500toInf.root"],
			"T-tchan": [Skimmed_Ganesh_base + "ST_t-channel_top_4f_InclusiveDecays.root"],
			"Tbar-tchan": [Skimmed_Ganesh_base + "ST_t-channel_antitop_4f_InclusiveDecays.root"],
			"T-tW": [Skimmed_Ganesh_base + "ST_tW_top_5f_inclusiveDecays.root"],
			"Tbar-tW": [Skimmed_Ganesh_base + "ST_tW_antitop_5f_inclusiveDecays.root"],
			"ST_s-channel_4f_hadronicDecays": [Skimmed_Ganesh_base + "ST_s-channel_4f_hadronicDecays.root"],
			"ST_s-channel_4f_leptonDecays": [Skimmed_Ganesh_base + "ST_s-channel_4f_leptonDecays.root"],
			"WJetsToLNu_HT-70To100": [Skimmed_Ganesh_base + "WJetsToLNu_HT-70To100.root"],
			"WJetsToLNu_HT-100To200": [Skimmed_Ganesh_base + "WJetsToLNu_HT-100To200.root"],
			"WJetsToLNu_HT-200To400": [Skimmed_Ganesh_base + "WJetsToLNu_HT-200To400.root"],
			"WJetsToLNu_HT-400To600": [Skimmed_Ganesh_base + "WJetsToLNu_HT-400To600.root",
				Skimmed_Ganesh_base +"WJetsToLNu_HT-400To600_2.root"],
			"WJetsToLNu_HT-600To800": [Skimmed_Ganesh_base + "WJetsToLNu_HT-600To800.root",
				Skimmed_Ganesh_base + "WJetsToLNu_HT-600To800_2.root"],
			"WJetsToLNu_HT-800To1200": [Skimmed_Ganesh_base + "WJetsToLNu_HT-800To1200.root",
				Skimmed_Ganesh_base + "WJetsToLNu_HT-800To1200_2.root"],
			"WJetsToLNu_HT-1200To2500": [Skimmed_Ganesh_base + "WJetsToLNu_HT-1200To2500.root",
				Skimmed_Ganesh_base + "WJetsToLNu_HT-1200To2500_2.root"],
			"WJetsToLNu_HT-2500ToInf": [Skimmed_Ganesh_base + "WJetsToLNu_HT-2500ToInf.root",
				Skimmed_Ganesh_base + "WJetsToLNu_HT-2500ToInf_2.root"],
			#QCD Samples
			#"QCD_HT50to100": [Skimmed_Ganesh_base + "QCD_HT50to100.root"], "QCD_HT100to200": [Skimmed_Ganesh_base + "QCD_HT100to200.root"], 
			#"QCD_HT200to300": [Skimmed_Ganesh_base + "QCD_HT200to300.root"], "QCD_HT300to500": [Skimmed_Ganesh_base + "QCD_HT300to500.root"],
			#"QCD_HT500to700": [Skimmed_Ganesh_base + "QCD_HT500to700.root"], "QCD_HT700to1000": [Skimmed_Ganesh_base + "QCD_HT700to1000.root"],
			#"QCD_HT1000to1500": [Skimmed_Ganesh_base + "QCD_HT1000to1500.root"], "QCD_HT1500to2000": [Skimmed_Ganesh_base + "QCD_HT1500to2000.root"],
			#"QCD_HT2000toInf": [Skimmed_Ganesh_base + "QCD_HT2000toInf.root"],
			"Data_Mu": [Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018A.root",Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018B.root",Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018C.root",
				Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018D.root",Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018D_2.root",Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018D_3.root",
				Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018D_4.root"]
			#"Data_MET": [Skimmed_Ganesh_base + "MET/MET_Run2018A.root",Skimmed_Ganesh_base + "MET/MET_Run2018B.root",Skimmed_Ganesh_base + "MET/MET_Run2018C.root",
			#	Skimmed_Ganesh_base + "MET/MET_Run2018D.root",Skimmed_Ganesh_base + "MET/MET_Run2018D_2.root",Skimmed_Ganesh_base + "MET/MET_Run2018D_3.root",
			#	Skimmed_Ganesh_base + "MET/MET_Run2018D_4.root"]
		}
	
	#Background lists 
	background_list_full = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"] #Full background list
	background_list_test = [r"$ZZ \rightarrow 4l$"] #Only ZZ4l background for testing
	background_list_none = [] #No backgrounds for data only testing
	
	#Set file dictionary and list of backgrounds prior to running processor
	file_dict = file_dict_full
	#file_dict = file_dict_data_test
	
	numEvents_Dict = {}
	sumWEvents_Dict = {}

	start_time = time.time()
	for key_name, file_array in file_dict.items(): 
		print(key_name)
		if (key_name != "Data_Mu" and key_name != "Data_MET" ): 
			numEvents_Dict[key_name] = 0 #Initialize the number of events dictionary
			sumWEvents_Dict[key_name] = 0 #Initialize the number of events dictionary
			for file in file_array:
				with uproot.open(file) as tempFile:
					print(file)
					numEvents_Dict[key_name] += np.sum(tempFile['Runs/genEventCount'].array()) #Fixed for nanoAOD 
					sumWEvents_Dict[key_name] += np.sum(tempFile['Runs/genEventSumw'].array()) #Fixed for nanoAOD 
					print(key_name + "sum: %f"%numEvents_Dict[key_name])

		else: #Ignore data files
			numEvents_Dict[key_name] = 1
			sumWEvents_Dict[key_name] = 1

	#Background names for single background plot file names
	background_plot_names = {r"$t\bar{t}$" : "_ttbar_", r"$t\bar{t}$ Hadronic" : "_ttbarHadronic_", r"$t\bar{t}$ Semileptonic" : "_ttbarSemilepton_",
			r"$t\bar{t}$ 2L2Nu" : "_ttbar2L2Nu_", r"Drell-Yan+Jets": "_DYJets_", "Di-Bosons" : "_DiBosons_", "Single Top": "_SingleTop_", "QCD" : "_QCD_", 
			"W+Jets" : "_WJets_", r"$ZZ \rightarrow 4l$" : "_ZZ4l_", r"$ZZ \rightarrow 4l$ Test": "_ZZ4lTest_", r"$ZZ \rightarrow 4l$ Control": "_ZZ4lControl_",
			"W+Jets HT 70-100 GeV" : "_WJetsHT70-100_","W+Jets HT 100-200 GeV" : "_WJetsHT100-200_","W+Jets HT 200-400 GeV" : "_WJetsHT200-400_",
			"W+Jets HT 400-600 GeV" : "_WJetsHT400-600_","W+Jets HT 600-800 GeV" : "_WJetsHT600-800_","W+Jets HT 800-1200 GeV" : "_WJetsHT800-1200_",
			"W+Jets HT 1200-2500 GeV" : "_WJetsHT1200-2500_","W+Jets HT 2500-Inf GeV" : "_WJetsHT2500-Inf_"} #For file names
	
	#Background names to samples dictionary
	background_dict = {r"$t\bar{t}$" : ["TTToSemiLeptonic","TTTo2L2Nu","TTToHadronic"], r"$t\bar{t}$ Hadronic" : ["TTToHadronic"], 
			r"$t\bar{t}$ Semileptonic" : ["TTToSemiLeptonic"], r"$t\bar{t}$ 2L2Nu" : ["TTTo2L2Nu"],
			r"Drell-Yan+Jets": ["DYJetsToLL_M-4to50_HT-70to100","DYJetsToLL_M-4to50_HT-100to200","DYJetsToLL_M-4to50_HT-200to400","DYJetsToLL_M-4to50_HT-400to600",
			"DYJetsToLL_M-4to50_HT-600toInf","DYJetsToLL_M-50_HT-70to100","DYJetsToLL_M-50_HT-100to200","DYJetsToLL_M-50_HT-200to400",
			"DYJetsToLL_M-50_HT-400to600","DYJetsToLL_M-50_HT-600to800","DYJetsToLL_M-50_HT-800to1200","DYJetsToLL_M-50_HT-1200to2500","DYJetsToLL_M-50_HT-2500toInf"], 
			"Di-Bosons": ["WZ2l2q","WZ1l1nu2q","ZZ2l2q", "WZ1l3nu", "VV2l2nu", "WWTo1L1Nu2Q", "WWTo4Q", "ZZTo4Q", "ZZTo2L2Nu", "ZZTo2Nu2Q"], 
			"Single Top": ["Tbar-tchan","T-tchan","Tbar-tW","T-tW","ST_s-channel_4f_leptonDecays", "ST_s-channel_4f_hadronicDecays"], 
			"W+Jets": ["WJetsToLNu_HT-70To100","WJetsToLNu_HT-100To200","WJetsToLNu_HT-200To400","WJetsToLNu_HT-400To600","WJetsToLNu_HT-600To800","WJetsToLNu_HT-800To1200","WJetsToLNu_HT-1200To2500","WJetsToLNu_HT-2500ToInf"],
			"W+Jets HT 100-200 GeV": ["WJetsToLNu_HT-100To200"],"W+Jets HT 200-400 GeV": ["WJetsToLNu_HT-200To400"],"W+Jets HT 400-600 GeV": ["WJetsToLNu_HT-400To600"],
			"W+Jets HT 600-800 GeV": ["WJetsToLNu_HT-600To800"],"W+Jets HT 800-1200 GeV": ["WJetsToLNu_HT-800To1200"],
			"W+Jets HT 1200-2500 GeV": ["WJetsToLNu_HT-1200To2500"], "W+Jets HT 2500-Inf GeV": ["WJetsToLNu_HT-2500ToInf"],
			r"$ZZ \rightarrow 4l$" : ["ZZ4l"],
			"QCD": ["QCD_HT50to100","QCD_HT100to200","QCD_HT200to300","QCD_HT300to500","QCD_HT500to700","QCD_HT700to1000","QCD_HT1000to1500","QCD_HT1500to2000","QCD_HT2000toInf"],
	}

	#Dictinary with file names
	trigger_name = "Mu_Trigger_WithQCD"
	four_tau_names = {
		"boostedtau_pt_Trigg": "BoostedTau_pT_Trigger" + "-" + trigger_name,
		"Leadingboostedtau_pt_Trigg": "BoostedTau_Leading_pT_Trigger" + "-" + trigger_name,
		"Subleadingboostedtau_pt_Trigg": "BoostedTau_Subleading_pT_Trigger" + "-" + trigger_name, 
		"Thirdleadingboostedtau_pt_Trigg": "BoostedTau_3rdleading_pT_Trigger" + "-" + trigger_name, 
		"Fourthleadingboostedtau_pt_Trigg": "BoostedTau_4thleading_pT_Trigger" + "-" + trigger_name, 
		"boostedtau_eta_Trigg": "BoostedTau_eta_Trigger" + "-" + trigger_name,
		"boostedtau_phi_Trigg": "BoostedTau_phi_Trigger" + "-" + trigger_name,
		"boostedtau_iso_Trigg": "BoostedTau_iso_Trigger" + "-" + trigger_name,
		"tau_pt_Trigg": "Tau_pT_Trigger" + "-" + trigger_name,
		"Leadingtau_pt_Trigg": "Tau_Leading_pT_Trigger" + "-" + trigger_name,
		"tau_eta_Trigg": "Tau_eta_Trigger" + "-" + trigger_name,
		"tau_phi_Trigg": "Tau_phi_Trigger" + "-" + trigger_name,
		"electron_pt_Trigg": "Electron_pT_Trigger" + "-" + trigger_name,
		"Leadingelectron_pt_Trigg": "Electron_Leading_pT_Trigger" + "-" + trigger_name,
		"electron_eta_Trigg": "Electron_eta_Trigger" + "-" + trigger_name,
		"electron_phi_Trigg": "Electron_phi_Trigger" + "-" + trigger_name,
		"muon_pt_Trigg": "Muon_pT_Trigger" + "-" + trigger_name,
		"Leadingmuon_pt_Trigg": "Muon_Leading_pT_Trigger" + "-" + trigger_name,
		"muon_eta_Trigg": "Muon_eta_Trigger" + "-" + trigger_name,
		"muon_phi_Trigg": "Muon_phi_Trigger" + "-" + trigger_name,
		"Jet_pt_Trigg": "Jet_pT_Trigger" + "-" + trigger_name,
		"LeadingJet_pt_Trigg": "Jet_Leading_pT_Trigger" + "-" + trigger_name,
		"Jet_eta_Trigg": "Jet_eta_Trigger" + "-" + trigger_name,
		"Jet_phi_Trigg": "Jet_phi_Trigger" + "-" + trigger_name,
		"AK8Jet_pt_Trigg": "AK8Jet_pT_Trigger" + "-" + trigger_name,
		"LeadingAK8Jet_pt_Trigg": "AK8Jet_Leading_pT_Trigger" + "-" + trigger_name,
		"AK8Jet_eta_Trigg": "AK8Jet_eta_Trigger" + "-" + trigger_name,
		"AK8Jet_phi_Trigg": "AK8Jet_phi_Trigger" + "-" + trigger_name,
		"MET": "MET_Trigger" + "-" + trigger_name,
		"HT": "HT_Trigger" + "-" + trigger_name,
		"MHT": "MHT_Trigger" + "-" + trigger_name,
	}
	
	for n_taus in range(4,5):
		start_time = time.time()
		print("About to run processor")
		fourtau_out = runner(file_dict, treename="Events", processor_instance=SkimProcessor.PlottingScriptProcessor(sumWEvents_Dict = sumWEvents_Dict, nBoostedTaus = n_taus, ApplyTrigger = False)) #Modified for NanoAOD (changd treename)
		end_time = time.time()
		
		time_running = end_time-start_time
		print("It takes about %.1f s to run the coffea processor with %d boosted tau selections"%(time_running,n_taus))
		
		#Save coffea file
		outfile = os.path.join(os.getcwd() + "/Output_2b2Tau/", f"output_{n_taus}_boosted_tau_selec_SingleMu2018A_MyCopyOfSkims.coffea")
		util.save(fourtau_out, outfile)
		print(f"Saved output to {outfile}")	
