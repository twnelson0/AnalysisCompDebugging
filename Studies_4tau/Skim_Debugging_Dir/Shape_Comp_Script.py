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
import sys
import argparse

#Plot style variables defined
hep.style.use(hep.style.CMS)
TABLEAU_COLORS = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

#Use arguement parser to handle command line arguemetns
#parse = argparse.ArgumentParser()
#parse.add_argument("-f", "--File", help = "Input coffea file")
#parse.add_argument("-n", "--NumberTau", help = "Number of boosted taus in selection")
#args = parse.parse_args()

if __name__ == "__main__":
	#coffea_file = "output_2018_run20260201_081729.coffea" #Store coffea file as hardcoded variable
	#coffea_file = "third_leading_boostedtau.coffea" #Store coffea file as hardcoded variable
	#coffea_file = "fourth_leading_boostedtau.coffea" #Store coffea file as hardcoded variable

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

	add_var = ["Leadingboostedtau_pt_Trigg", "Subleadingboostedtau_pt_Trigg","Thirdleadingboostedtau_pt_Trigg","Fourthleadingboostedtau_pt_Trigg"]
	four_tau_hist_list = add_var + four_tau_hist_list
	
	#Dictinary with file names
	trigger_name = "SingleMu_Trigger"
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
	coffea_input_4tau = util.load("output_4_boosted_tau_selec_SingleMu2018A_4tauSamples.coffea")
	coffea_input_2b2tau = util.load("output_4_boosted_tau_selec_SingleMu2018A_2b2tauSamples.coffea")

	#Print the amount of data in the 4tau samples
	print("=============================================")
	print("4 Tau Samples")
	print("Number of data events: %d"%coffea_input_4tau["Data_Mu"]["Event_Count"])
	print("=============================================")

	print("Number of events prior to selections: %d"%coffea_input_4tau["Data_Mu"]["n_Skim"])
	print("Number of events after MET selection: %d"%coffea_input_4tau["Data_Mu"]["n_MET"])
	print("Number of events after FatJet selection: %d"%coffea_input_4tau["Data_Mu"]["n_FatJet"])
	print("Number of events after quality flag selection: %d"%coffea_input_4tau["Data_Mu"]["n_FlagSelec"])
	print("Number of events after Primary Vertex selection: %d"%coffea_input_4tau["Data_Mu"]["n_PVSelec"])
	print("Number of events after Leading Boosted Tau selection: %d"%coffea_input_4tau["Data_Mu"]["n_LeadTau"])
	print("Number of events after Sub-Leading Boosted Tau selection: %d"%coffea_input_4tau["Data_Mu"]["n_SubLeadTau"])
	print("Number of events after 3rd-Leading Boosted Tau selection: %d"%coffea_input_4tau["Data_Mu"]["n_3rdLeadTau"])
	print("Number of events after 4th-Leading Boosted Tau selection: %d"%coffea_input_4tau["Data_Mu"]["n_4thLeadTau"])

	#Print the amount of data in the 2b2tau samples
	print("=============================================")
	print("2b2Tau Samples")
	print("Number of data events: %d"%coffea_input_2b2tau["Data_Mu"]["Event_Count"])
	print("=============================================")

	print("Number of events prior to selections: %d"%coffea_input_2b2tau["Data_Mu"]["n_Skim"])
	print("Number of events after MET selection: %d"%coffea_input_2b2tau["Data_Mu"]["n_MET"])
	print("Number of events after FatJet selection: %d"%coffea_input_2b2tau["Data_Mu"]["n_FatJet"])
	print("Number of events after quality flag selection: %d"%coffea_input_2b2tau["Data_Mu"]["n_FlagSelec"])
	print("Number of events after Primary Vertex selection: %d"%coffea_input_2b2tau["Data_Mu"]["n_PVSelec"])
	print("Number of events after Leading Boosted Tau selection: %d"%coffea_input_2b2tau["Data_Mu"]["n_LeadTau"])
	print("Number of events after Sub-Leading Boosted Tau selection: %d"%coffea_input_2b2tau["Data_Mu"]["n_SubLeadTau"])
	print("Number of events after 3rd-Leading Boosted Tau selection: %d"%coffea_input_2b2tau["Data_Mu"]["n_3rdLeadTau"])
	print("Number of events after 4th-Leading Boosted Tau selection: %d"%coffea_input_2b2tau["Data_Mu"]["n_4thLeadTau"])


	#Produce N-1 and cutflow plots for data
#	figcut, axcut = plt.subplots()
#	coffea_input["Data_Mu"]["Mini_Cutflow"].plot1d(ax = axcut)
#	plt.savefig("Data_Cutflow_Plot.png")
#    
#	figcut, axcut = plt.subplots()
#	coffea_input["Data_Mu"]["Mini_NMinus1"].plot1d(ax = axcut)
#	plt.savefig("Data_NMinus_Plot.png")

	#Prdouce histograms from the coffea file
	for hist_name in four_tau_hist_list: #Loop over all histograms
		#MPLHEP ratio plot
		if (hist_name == "Leadingmuon_eta_Trigg"):
			axis_label = r"Leading $\mu$ $\eta$"
		else:
			axis_label = coffea_input_4tau["Data_Mu"][hist_name].axes[0].label

		#Normalize Histograms
		hist_4tau = coffea_input_4tau["Data_Mu"][hist_name]*1/coffea_input_4tau["Data_Mu"][hist_name].sum()
		hist_2b2tau = coffea_input_2b2tau["Data_Mu"][hist_name]*1/coffea_input_2b2tau["Data_Mu"][hist_name].sum()
		
		#Produce histograms
		fig, ax_main, ax_comp = hep.comp.hists(
			h1 = hist_4tau,
			h2 = hist_2b2tau,
			xlabel = axis_label,
			comparison = "ratio",
            markersize = 5,
			h1_label = "4Tau Samples",
			h2_label = "2b2Tau Samples",
		)
		hep.cms.label(data=True, ax = ax_main, text = "Single Muon 2018A Data")	
		plt.savefig(four_tau_names[hist_name] + "_" + "CompDist")
		plt.close()


