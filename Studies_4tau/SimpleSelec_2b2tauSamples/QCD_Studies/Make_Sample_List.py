#Import modules
import numpy as np

if __name__ == "__main__":
	#Diretory for files
	Skimmed_Ganesh_base = "/hdfs/store/user/gparida/HHbbtt/Hadded_Skimmed_Files/Full_Production_CMSSW_13_0_13_Nov24_23/LooseSelection_MET_gt_80_nFatJet_gt_0_Skim/2018/"
	
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
			"QCD_HT50to100": [Skimmed_Ganesh_base + "QCD_HT50to100.root"], "QCD_HT100to200": [Skimmed_Ganesh_base + "QCD_HT100to200.root"], 
			"QCD_HT200to300": [Skimmed_Ganesh_base + "QCD_HT200to300.root"], "QCD_HT300to500": [Skimmed_Ganesh_base + "QCD_HT300to500.root"],
			"QCD_HT500to700": [Skimmed_Ganesh_base + "QCD_HT500to700.root"], "QCD_HT700to1000": [Skimmed_Ganesh_base + "QCD_HT700to1000.root"],
			"QCD_HT1000to1500": [Skimmed_Ganesh_base + "QCD_HT1000to1500.root"], "QCD_HT1500to2000": [Skimmed_Ganesh_base + "QCD_HT1500to2000.root"],
			"QCD_HT2000toInf": [Skimmed_Ganesh_base + "QCD_HT2000toInf.root"],
			"Data_Mu": [Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018A.root",Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018B.root",Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018C.root",
				Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018D.root",Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018D_2.root",Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018D_3.root",
				Skimmed_Ganesh_base + "SingleMu/SingleMu_Run2018D_4.root"]
	}

	background_list_full_QCD = [r"$t\bar{t}$", r"Drell-Yan+Jets", "Di-Bosons", "Single Top", "W+Jets", r"$ZZ \rightarrow 4l$","QCD","Data_Mu"] #Full background list (with QCD)

	#for background in background_list_full_QCD:
	for sample in file_dict_full.keys():
		for file in file_dict_full[sample]:
			print(file)
