import awkward as ak
import uproot
import hist
from hist import intervals
import matplotlib as mpl
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
import sys
import argparse
import itertools

#Plot style variables defined
hep.style.use(hep.style.CMS)
#cmap = mpl.colormaps['PiYG'] 
#cmap = mpl.colormaps['plasma'] 
cmap = mpl.colormaps['hsv'] 

#Collection of color maps
cmap0 = mpl.colormaps['Reds']
cmap1 = mpl.colormaps['Greens']
cmap2 = mpl.colormaps['Blues']
cmap3 = mpl.colormaps['Oranges']
cmap4 = mpl.colormaps['Greys']
cmap5 = mpl.colormaps['Purples']
cmap_array = [cmap0,cmap1,cmap2,cmap3,cmap4,cmap5]

TABLEAU_COLORS = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

#Use arguement parser to handle command line arguemetns
parse = argparse.ArgumentParser()
parse.add_argument("-f", "--File", help = "Input coffea file")
parse.add_argument("-n", "--NumberTau", help = "Number of boosted taus in selection")
args = parse.parse_args()

if __name__ == "__main__":
	#coffea_file = "output_2018_run20260201_081729.coffea" #Store coffea file as hardcoded variable
	#coffea_file = "third_leading_boostedtau.coffea" #Store coffea file as hardcoded variable
	#coffea_file = "fourth_leading_boostedtau.coffea" #Store coffea file as hardcoded variable

	print("Running on file " + args.File)
	print("With " + args.NumberTau + " Boosted taus required")

	coffea_file = args.File

	#Dictionaries and arrays with information on plot constrution, naming and samples
	four_tau_hist_list = [
			"boostedtau_pt_Trigg","boostedtau_eta_Trigg","boostedtau_phi_Trigg",
			"tau_pt_Trigg","tau_eta_Trigg","tau_phi_Trigg",
			"electron_pt_Trigg","electron_eta_Trigg","electron_phi_Trigg",
			"muon_pt_Trigg","muon_eta_Trigg","muon_phi_Trigg","Leadingmuon_pt_Trigg","Leadingmuon_eta_Trigg",
			"Jet_pt_Trigg","Jet_eta_Trigg","Jet_phi_Trigg",
			"AK8Jet_pt_Trigg","AK8Jet_eta_Trigg","AK8Jet_phi_Trigg",
			"MET","HT","MHT" #, "Mini_Cutflow", "Mini_NMinus1"
			]

	#Additional boosted tau distributions to pull based on boosted tau requirements
	add_var = []
	n_tau = int(args.NumberTau)
	if (n_tau == 1):
		print("One Boosted tau required")
		add_var = ["Leadingboostedtau_pt_Trigg"]
	if (n_tau == 2):
		print("Two Boosted taus required")
		add_var = ["Leadingboostedtau_pt_Trigg", "Subleadingboostedtau_pt_Trigg"]
	if (n_tau == 3):
		print("Three Boosted taus required")
		add_var = ["Leadingboostedtau_pt_Trigg", "Subleadingboostedtau_pt_Trigg","Thirdleadingboostedtau_pt_Trigg"]
	if (n_tau == 4):
		print("Four Boosted taus required")
		add_var = ["Leadingboostedtau_pt_Trigg", "Subleadingboostedtau_pt_Trigg","Thirdleadingboostedtau_pt_Trigg","Fourthleadingboostedtau_pt_Trigg"]

	four_tau_hist_list = add_var + four_tau_hist_list
	
	background_list_full = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"]
	background_list_QCD = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$","QCD"]
	background_list_test = [r"$ZZ \rightarrow 4l$"]
	background_list_none = []
	background_list = background_list_test
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
	
	all_sample_array = []
	for background in background_list:
		all_sample_array.append(background_dict[background])
	
	#Make the colors easier on the eyes
	all_sample_array = list(itertools.chain.from_iterable(all_sample_array))
	background_list = all_sample_array #Produce list of all backgrounds rather than combining them
	#TABLEAU_COLORS = []
	
	#for background_type in background_dict.keys():
#	for i in range(len(background_list_full)):
#		TABLEAU_COLORS.append(cmap_array[i](np.linspace(0,1,len(background_dict[background_list_full[i]]))))
	TABLEAU_COLORS = cmap(np.linspace(0,1,len(background_list)))

	#print(TABLEAU_COLORS)
	
	#Dictinary with file names
	trigger_name = "SingleMu_Trigger"
	four_tau_names = {
		"boostedtau_pt_Trigg": "BoostedTau_pT_Trigger" + "-" + trigger_name,
		#"Leadingboostedtau_pt_Trigg": "BoostedTau_Leading_pT_Trigger" + "-" + trigger_name,
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
		"Leadingmuon_eta_Trigg": "Muon_Leading_eta_Trigger" + "-" + trigger_name,
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
		"Mini_Cutflow": "Mini_Cutflow_Trigger" + "-" + trigger_name,
		"Mini_NMinus1": "Mini_NMinus1_Trigger" + "-" + trigger_name,
	}


	#Import coffea files with histograms
	coffea_input = util.load(coffea_file)

	
	print("Cutflow of ZZ4l")
	print("Number of events prior to selections: %d"%coffea_input["ZZ4l"]["n_Skim"])
	print("Number of events after HLT Trigger Selection: %d"%coffea_input["ZZ4l"]["n_HLT"])
	print("Number of events after nMuon Selection: %d"%coffea_input["ZZ4l"]["n_nMuon"])
	print("Number of events after Offline Muon Selection: %d"%coffea_input["ZZ4l"]["n_OfflineMu"])
	print("Number of events after Trigger (offline + HLT): %d"%coffea_input["ZZ4l"]["n_Trigger"])
	print("Number of events after Leading Boosted Tau selection: %d"%coffea_input["ZZ4l"]["n_LeadBoostedTau"])
	print("Number of events after Sub-Leading Boosted Tau selection: %d"%coffea_input["ZZ4l"]["n_SubLeadBoostedTau"])
	print("Number of events after 3rd-Leading Boosted Tau selection: %d"%coffea_input["ZZ4l"]["n_3rdLeadBoostedTau"])
	print("Number of events after 4th-Leading Boosted Tau selection: %d"%coffea_input["ZZ4l"]["n_4thLeadBoostedTau"])
	print("Number of events after MET selection: %d"%coffea_input["ZZ4l"]["n_MET"])
	print("Number of events after FatJet selection: %d"%coffea_input["ZZ4l"]["n_FatJet"])
	print("Number of events after quality flag selection: %d"%coffea_input["ZZ4l"]["n_FlagSelec"])
	print("Number of events after Primary Vertex selection: %d"%coffea_input["ZZ4l"]["n_PVSelec"])



	#Produce N-1 and cutflow plots for data
	figcut, axcut = plt.subplots()
	coffea_input["ZZ4l"]["Mini_Cutflow"].plot1d(ax = axcut)
	plt.savefig("Data_Cutflow_Plot.png")
    
	figcut, axcut = plt.subplots()
	coffea_input["ZZ4l"]["Mini_NMinus1"].plot1d(ax = axcut)
	plt.savefig("Data_NMinus_Plot.png")
	
	#Dictionaries of histograms for background, signal and data
	hist_dict_background = dict.fromkeys(four_tau_hist_list)
	hist_dict_signal = dict.fromkeys(four_tau_hist_list)
	hist_dict_data = dict.fromkeys(four_tau_hist_list)



