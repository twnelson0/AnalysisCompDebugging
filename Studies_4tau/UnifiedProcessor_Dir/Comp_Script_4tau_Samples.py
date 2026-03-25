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
				#"+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest-py3.10"',
				#"Requirements": "HasSingularityJobStart",
                "container_image": "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-cc7:latest-py3.10",
				"InitialDir": f'/scratch/{os.environ["USER"]}',
				'transfer_input_files': f"{_x509_path}",

			},
			job_script_prologue = [
				"export XRD_RUNFORKHANDLER=1",
				f"export X509_USER_PROXY={_x509_path}",
			]
	)
	cluster.adapt(minimum=1, maximum=500)
	
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
		print("Run Iteratively")
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
	Skimmed_4tau_base_MC = "root://cmsxrootd.hep.wisc.edu//store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/"
	Skimmed_4tau_base_Data = "root://cmsxrootd.hep.wisc.edu//store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/Data/"
	Skimmed_4tau_loc_Data = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/Data/"
	Skimmed_4tau_loc_MC = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/"

	#Make full arrays of single Muon data
	SingleMu_2018A = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root") 
	SingleMu_2018B = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018B_15January26_0731_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root") 
	SingleMu_2018C = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018C_15January26_0740_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root") 
	SingleMu_2018D = glob.glob(Skimmed_4tau_loc_Data + "SingleMu_Run2018D_15January26_0815_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root") 

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

	file_dict_data_test = {
		#"Data_Mu" : [Skimmed_4tau_base_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim/SingleMu_Run2018A.root"]
		#"Data_Mu": [Skimmed_4tau_base_Data + "SingleMu_Run2018A_15January26_0751_skim_Jan26Skim/singleFileSkimForSubmission-NANO_NANO_*.root"]
		"Data_Mu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in SingleMu_2018A_Debug],
		#"Data_Mu": [file for file in SingleMu_2018A_Debug] 
		#"Data_Mu": [] 
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
			"Data_Mu": ["root://cmsxrootd.hep.wisc.edu//" + file[6:] for file in np.append(SingleMu_2018A, np.append(SingleMu_2018B, np.append(SingleMu_2018C,SingleMu_2018D)))]
		}
	
	#Background lists 
	background_list_full = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$","QCD"] #Full background list (with QCD)
	background_list_full = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"] #Full background list
	background_list_test = [r"$ZZ \rightarrow 4l$"] #Only ZZ4l background for testing
	background_list_none = [] #No backgrounds for data only testing
	
	#Set file dictionary and list of backgrounds prior to running processor
	#file_dict = file_dict_data_test
	file_dict = file_dict_full

	#Pull in the weight and event count prior to skimming information
	with open("genWeightSum_JSON.json") as json_file:
		sumWEvents_Dict = json.load(json_file)

	with open("numEvents_JSON.json") as json_file:
		numEvents_Dict = json.load(json_file)
	

	start_time = time.time()

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
		print("About to run processor")
		start_time = time.time()
		fourtau_out = runner(file_dict, treename="Events", processor_instance=SkimProcessor.PlottingScriptProcessor(sumWEvents_Dict = sumWEvents_Dict, nBoostedTaus = n_taus, ApplyTrigger = False)) #Modified for NanoAOD (changd treename)
		end_time = time.time()
		
		time_running = end_time-start_time
		print("It takes about %.1f s to run the coffea processor with %d boosted tau selections"%(time_running,n_taus))
		
        #Save coffea file
		outfile = os.path.join(os.getcwd() + "/Output_4Tau/", f"output_{n_taus}_boosted_tau_selec_SingleMu2018A.coffea")
		util.save(fourtau_out, outfile)
		print(f"Saved output to {outfile}")	
