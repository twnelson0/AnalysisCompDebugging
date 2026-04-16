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
	
	background_list_full = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$"] #,"QCD"]
	background_list_test = [r"$ZZ \rightarrow 4l$"]
	background_list_none = []
	background_list = background_list_full
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
			#"QCD": ["QCD_HT50to100","QCD_HT100to200","QCD_HT200to300","QCD_HT300to500","QCD_HT500to700","QCD_HT700to1000","QCD_HT1000to1500","QCD_HT1500to2000","QCD_HT2000toInf"],
	}
	
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

	#Print the abount of data
	print("=============================================")
	print("Number of data events after all selections: %d"%coffea_input["Data_MET"]["Event_Count"])
	print("=============================================")

	print("Number of events prior to selections: %d"%coffea_input["Data_MET"]["n_Skim"])
	print("Number of events after MET selection: %d"%coffea_input["Data_MET"]["n_MET"])
	print("Number of events after FatJet selection: %d"%coffea_input["Data_MET"]["n_FatJet"])
	print("Number of events after quality flag selection: %d"%coffea_input["Data_MET"]["n_FlagSelec"])
	print("Number of events after Primary Vertex selection: %d"%coffea_input["Data_MET"]["n_PVSelec"])
	print("Number of events after Leading Boosted Tau selection: %d"%coffea_input["Data_MET"]["n_LeadBoostedTau"])
	print("Number of events after Sub-Leading Boosted Tau selection: %d"%coffea_input["Data_MET"]["n_SubLeadBoostedTau"])
	print("Number of events after 3rd-Leading Boosted Tau selection: %d"%coffea_input["Data_MET"]["n_3rdLeadBoostedTau"])
	print("Number of events after 4th-Leading Boosted Tau selection: %d"%coffea_input["Data_MET"]["n_4thLeadBoostedTau"])
	print("Number of events after Trigger: %d"%coffea_input["Data_MET"]["n_Trigger"])


	#Produce N-1 and cutflow plots for data
	figcut, axcut = plt.subplots()
	coffea_input["Data_MET"]["Mini_Cutflow"].plot1d(ax = axcut)
	plt.savefig("Data_Cutflow_Plot.png")
    
	figcut, axcut = plt.subplots()
	coffea_input["Data_MET"]["Mini_NMinus1"].plot1d(ax = axcut)
	plt.savefig("Data_NMinus_Plot.png")
	
	#Dictionaries of histograms for background, signal and data
	hist_dict_background = dict.fromkeys(four_tau_hist_list)
	hist_dict_signal = dict.fromkeys(four_tau_hist_list)
	hist_dict_data = dict.fromkeys(four_tau_hist_list)

	#Prdouce histograms from the coffea file
	for hist_name in four_tau_hist_list: #Loop over all histograms

		temp_hist_dict = dict.fromkeys(background_list) # create dictionary of histograms for each background type
				
		for background_type in background_list:
			#print("Background type %s"%background_type)
			background_array = []
			backgrounds = background_dict[background_type]
						
			#Loop over all backgrounds
			for background in backgrounds:
				#print("%s"%background)
				if (True): #Only need to generate single background once
					
					#Plot the cutflow for each background
					if (hist_name == "cutflow_table"):
						#print(coffea_input[background]["cutflow_table"].axes)
						if (background == backgrounds[0]):
							cutflow_hist = coffea_input[background]["cutflow_table"]
						else:
							cutflow_hist += coffea_input[background]["cutflow_table"]
							
					#	if (background == backgrounds[-1]):
					#		fig2p5, ax2p5 = plt.subplots()
					#		cutflow_hist.plot1d(ax=ax2p5)
					#		plt.title(background_type + " Cutflow Table")
					#		ax2p5.set_yscale('log')
					#		plt.savefig("SingleBackground" + background_plot_names[background_type] + "CutFlowTable")
					#		plt.close()
							
						#Plot the weights for each background
						if (background == backgrounds[0]):
							weight_hist = coffea_input[background]["weight_Hist"]
						else:
							weight_hist += coffea_input[background]["weight_Hist"]
						if (background == backgrounds[-1]):
							figweight, axweight = plt.subplots()
							weight_hist.plot1d(ax=axweight)
							plt.title(background_type + " Weight Histogram")
							plt.savefig("SingleBackground" + background_plot_names[background_type] + "Weight")
							plt.close()
							
					if (hist_name == "Radion_Charge_Arr"):
						lumi_table_data["MC Sample"].append(background)
						lumi_table_data["Luminosity"].append(coffea_input[background]["Lumi_Val"])
						lumi_table_data["Cross Section (pb)"].append(coffea_input[background]["CrossSec_Val"])
						#lumi_table_data["Number of Events"].append(coffea_input[background]["NEvent_Val"])
						lumi_table_data["Gen SumW"].append(coffea_input[background]["SumWEvent_Val"])
						lumi_table_data["Calculated Weight"].append(coffea_input[background]["Weight_Val"])
							
					if (hist_name != "Electron_tau_dR_Arr" and hist_name != "Muon_tau_dR_Arr"):
						if (background == backgrounds[0]):
							crnt_hist = coffea_input[background][hist_name]
							#print("Background: " + background)
							#print("Sum of entries: %f"%coffea_input[background][hist_name].sum())
						else:
							crnt_hist += coffea_input[background][hist_name]
							#print("Background: " + background)
							#print("Sum of entries: %f"%coffea_input[background][hist_name].sum())
						if (background == backgrounds[-1]):
							#fig2, ax2 = plt.subplots()
							temp_hist_dict[background_type] = crnt_hist #Try to fix stacking bug
							#crnt_hist.plot1d(ax=ax2)
							#if (hist_name == "FourTau_Mass_Arr"):
							#print("Background: " + background_type)
							#print("Sum of entries: %f"%crnt_hist.sum())
							#print("Number of Entries: %d"%coffea_input[background]["num_events"])
							#plt.title(background_type)
							#plt.savefig("SingleBackground" + background_plot_names[background_type] + four_tau_names[hist_name])
							#plt.close()

					else: #lepton-tau delta R 
					#	fig2, ax2 = plt.subplots()
						coffea_input[background][hist_name].plot1d(ax=ax2)
					#	ax2.set_yscale('log')
					#	plt.title(background_type)
					#	plt.savefig("SingleBackground" + background_plot_names[background_type] + four_tau_names[hist_name])
					#	plt.close()
						

		#Combine the backgrounds together
		hist_dict_background[hist_name] = hist.Stack.from_dict(temp_hist_dict) #This could be causing the problems 
		
		#Obtain data distributions
		#print("==================Hist %s================"%hist_name)
		hist_dict_data[hist_name] = coffea_input["Data_MET"][hist_name] #.fill("Data",coffea_input["Data_SingleMuon"][hist_name]) 
		background_stack = hist_dict_background[hist_name] #hist_dict_background[hist_name].stack("background")
		#signal_stack = hist_dict_signal[hist_name].stack("signal")
		
		data_stack = hist_dict_data[hist_name] #.stack("data")    
		#signal_array = [signal_stack["Signal"]]
		data_array = [data_stack] #["Data"]]
				
		for background in background_list:
			background_array.append(background_stack[background]) #Is this line fucking up your scaling??
			#print("Background: " + background)
			#print("Sum of stacked histogram: %f"%background_stack[background].sum())
		
		#MPLHEP ratio plot
		if (hist_name == "Leadingmuon_eta_Trigg"):
			axis_label = r"Leading $\mu$ $\eta$"
		else:
			axis_label = coffea_input["Data_MET"][hist_name].axes[0].label
		fig, ax_main, ax_comp = hep.comp.data_model(
			data_hist = coffea_input["Data_MET"][hist_name],
			stacked_components = background_array,
			stacked_colors = TABLEAU_COLORS[:len(background_list)],
			stacked_labels = background_list,
			xlabel = axis_label,
			model_uncertainty=True,
			comparison = "ratio",
            markersize = 10,
			#linewidth=2,

		)
		hep.cms.label(data=True, ax = ax_main, text = "2018 Data Preliminary")	
		plt.savefig(four_tau_names[hist_name] + "_" + str(args.NumberTau) + "TauSelec")
		plt.close()

					
		#Stack background distributions and plot signal + data distribution
#		fig,ax = plt.subplots()
#		hep.histplot(background_array,ax=ax,stack=True,histtype="fill",label=background_list,facecolor=TABLEAU_COLORS[:len(background_list)],edgecolor=TABLEAU_COLORS[:len(background_list)])
#		#hep.histplot(signal_array,ax=ax,stack=True,histtype="step",label=signal_list,edgecolor=TABLEAU_COLORS[len(background_list)+1],linewidth=2.95)
#		hep.histplot(data_array,ax=ax,stack=False,histtype="errorbar", yerr=True,label=["Data"],marker="o",color = "k") #,facecolor='black',edgecolor='black') #,mec='k')
#		hep.cms.text("Preliminary",loc=0,fontsize=13)
#		ax.set_title("2018 Data",loc = "right")
#		ax.legend(fontsize=10, loc='upper right')
#		plt.savefig(four_tau_names[hist_name])
#		plt.close()


