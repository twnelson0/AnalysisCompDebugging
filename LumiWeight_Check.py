import numpy as np
import matplotlib.pyplot as plt
import csv
import awkward as ak
import uproot

#Dictionary of cross sections 
xSection_Dictionary = {"Signal": 0.01, #Chosen to make plots readable
	#TTBar Background
	#"TTTo2L2Nu": 831.76*TT_FullLep_BR, "TTToSemiLeptonic": 831.76*TT_SemiLep_BR, "TTToHadronic": 831.76*TT_Had_BR,
	"TTTo2L2Nu": 87.5595, "TTToSemiLeptonic": 365.2482, "TTToHadronic": 381.0923,
	
	#DiBoson Background
	#"ZZ2l2q": 3.22, "WZ3l1nu": 4.708, "WZ2l2q": 5.595, "WZ1l1nu2q": 10.71, "VV2l2nu": 11.95, "WZ1l3nu": 3.05, #"WZ3l1nu.root" : 27.57,
	"ZZ2l2q": 3.676, "WZ2l2q": 6.565, "WZ1l1nu2q": 9.119, "WZ1l3nu": 3.414, "VV2l2nu": 11.09, #"WZ3l1nu.root" : 27.57,
	
	#ZZ->4l
	"ZZ4l": 1.325,
 
	"Tbar-tchan": 80.8, "T-tchan": 134.2, "Tbar-tW": 39.65, "T-tW": 39.65, 
	
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
	"WJetsToLNu_HT-100To200" : 1244.0, "WJetsToLNu_HT-200To400": 337.8, "WJetsToLNu_HT-400To600": 44.93, "WJetsToLNu_HT-600To800": 11.19, "WJetsToLNu_HT-800To1200": 4.926, "WJetsToLNu_HT-1200To2500" : 1.152, "WJetsToLNu_HT-2500ToInf" : 0.02646, 
	
	#SM Higgs
	"ZH125": 0.7544*0.0621, "ggZHLL125":0.1223 * 0.062 * 3 * 0.033658, "ggZHNuNu125": 0.1223*0.062*0.2,"ggZHQQ125": 0.1223*0.062*0.6991, "toptopH125": 0.5033*0.062, #"ggH125": 48.30* 0.0621, "qqH125": 3.770 * 0.0621, "WPlusH125": 
	#QCD
	"QCD_HT300to500": 347700, "QCD_HT500to700": 32100, "QCD_HT700to1000": 6831, "QCD_HT1000to1500": 1207, "QCD_HT1500to2000": 119.9, "QCD_HT2000toInf": 25.24,
}	
Lumi_2018 = 59830

def weight_calc(sample,numEvents=1):
	return Lumi_2018*xSection_Dictionary[sample]/numEvents

if __name__ == "__main__":
	#File locations
	my_background_base = "/hdfs/store/user/twnelson/HH4Tau_EtAl/Skimmed_Files/2018/MC/" 
	Ganesh_background_base = "/hdfs/store/user/gparida/HHbbtt/Framework_Processed_Files/Full_Production_CMSSW_13_0_13_Nov24_23/CommonAnalysis_6_WithSystematicScripts_April1_25/WeightAndSystematicsAndBoostedTauWts/2018/"

	my_background_dict = {
		"TTToSemiLeptonic": [my_background_base + "TTToSemiLeptonic_35August25_0448_skim_Newskim/TTToSemiLeptonic" + str(j) + ".root" for j in range(10)], 
		"TTTo2L2Nu": [my_background_base + "TTTo2L2Nu_26August25_0719_skim_Newskim/TTTo2L2Nu.root"], 
		"TTToHadronic": [my_background_base + "TTToHadronic_25October25_0813_skim_Newskim/TTToHadronic" + str(j) + ".root" for j in range(10)],
		"ZZ4l": [my_background_base + "ZZTo4L_26August25_0757_skim_Newskim/ZZTo4L.root"], 
		"VV2l2nu": [my_background_base + "WWTo2L2Nu_26August25_1040_skim_Newskim/WWTo2L2Nu.root"], 
		"WZ1l3nu": [my_background_base + "WZTo1L3Nu_4f_26August25_1016_skim_Newskim/WZTo1L3Nu_4f.root"], 
		"ZZ2l2q": [my_background_base + "ZZTo2Q2L_26August25_1034_skim_Newskim/ZZTo2Q2L.root"],
		"WZ2l2q": [my_background_base + "WZTo2L2Q_26August25_0926_skim_Newskim/WZTo2L2Q.root"],
		"WZ1l1nu2q" : [my_background_base + "WZTo1L1Nu2Q_26August25_0840_skim_Newskim/WZTo1L1Nu2Q.root"],
        #"DYJetsToLL_M-4to50_HT-70to100": [my_background_base + "DYJetsToLL_M-4to50_HT-70to100_12December25_0352_skim_Newskim/DYJetsToLL_M-4to50_HT-70to100.root"],
        #"DYJetsToLL_M-4to50_HT-100to200": [my_background_base + "DYJetsToLL_M-4to50_HT-100to200_12December25_0350_skim_Newskim/DYJetsToLL_M-4to50_HT-100to200.root"],
        "DYJetsToLL_M-4to50_HT-200to400": [my_background_base + "DYJetsToLL_M-4to50_HT-200to400_12December25_0325_skim_Newskim/DYJetsToLL_M-4to50_HT-200to400.root"],
        "DYJetsToLL_M-4to50_HT-400to600": [my_background_base + "DYJetsToLL_M-4to50_HT-400to600_12December25_0335_skim_Newskim/DYJetsToLL_M-4to50_HT-400to600.root"],
        "DYJetsToLL_M-4to50_HT-600toInf": [my_background_base + "DYJetsToLL_M-4to50_HT-600toInf_12December25_0354_skim_Newskim/DYJetsToLL_M-4to50_HT-600toInf.root"],
        "DYJetsToLL_M-50_HT-70to100": [my_background_base + "DYJetsToLL_M-50_HT-70to100_12December25_0340_skim_Newskim/DYJetsToLL_M-50_HT-70to100.root"],
        "DYJetsToLL_M-50_HT-100to200": [my_background_base + "DYJetsToLL_M-50_HT-100to200_12December25_0330_skim_Newskim/DYJetsToLL_M-50_HT-100to200.root"],
        "DYJetsToLL_M-50_HT-200to400": [my_background_base + "DYJetsToLL_M-50_HT-200to400_12December25_0344_skim_Newskim/DYJetsToLL_M-50_HT-200to400.root"],
        "DYJetsToLL_M-50_HT-400to600": [my_background_base + "DYJetsToLL_M-50_HT-400to600_12December25_0327_skim_Newskim/DYJetsToLL_M-50_HT-400to600.root"],
        "DYJetsToLL_M-50_HT-600to800": [my_background_base + "DYJetsToLL_M-50_HT-600to800_12December25_0339_skim_Newskim/DYJetsToLL_M-50_HT-600to800.root"],
        "DYJetsToLL_M-50_HT-800to1200": [my_background_base + "DYJetsToLL_M-50_HT-800to1200_12December25_0348_skim_Newskim/DYJetsToLL_M-50_HT-800to1200.root"],
        "DYJetsToLL_M-50_HT-1200to2500": [my_background_base + "DYJetsToLL_M-50_HT-1200to2500_12December25_0329_skim_Newskim/DYJetsToLL_M-50_HT-1200to2500.root"],
        "DYJetsToLL_M-50_HT-2500toInf": [my_background_base + "DYJetsToLL_M-50_HT-2500toInf_12December25_0338_skim_Newskim/DYJetsToLL_M-50_HT-2500toInf.root"],
		"T-tchan": [my_background_base + "ST_t-channel_top_4f_InclusiveDecays_26August25_0843_skim_Newskim/ST_t-channel_top_4f_InclusiveDecays.root"], 
		"Tbar-tchan": [my_background_base + "ST_t-channel_antitop_4f_InclusiveDecays_26August25_0821_skim_Newskim/ST_t-channel_antitop_4f_InclusiveDecays.root"], 
		"T-tW": [my_background_base + "ST_tW_top_5f_inclusiveDecays_26August25_0753_skim_Newskim/ST_tW_top_5f_inclusiveDecays.root"], 
		"Tbar-tW": [my_background_base + "ST_tW_antitop_5f_inclusiveDecays_26August25_1030_skim_Newskim/ST_tW_antitop_5f_inclusiveDecays.root"],
		"WJetsToLNu_HT-100To200": [my_background_base + "WJetsToLNu_HT-100To200_26August25_0810_skim_Newskim/WJetsToLNu_HT-100To200.root"],
		"WJetsToLNu_HT-200To400": [my_background_base + "WJetsToLNu_HT-200To400_26August25_0709_skim_Newskim/WJetsToLNu_HT-200To400.root"], 
		"WJetsToLNu_HT-400To600": [my_background_base + "WJetsToLNu_HT-400To600_26August25_1014_skim_Newskim/WJetsToLNu_HT-400To600.root", my_background_base +"WJetsToLNu_HT-400To600_OtherPart_26August25_1032_skim_Newskim/WJetsToLNu_HT-400To600_OtherPart.root"], 
		"WJetsToLNu_HT-600To800": [my_background_base + "WJetsToLNu_HT-600To800_26August25_0755_skim_Newskim/WJetsToLNu_HT-600To800.root", my_background_base + "WJetsToLNu_HT-600To800_OtherPart_26August25_0752_skim_Newskim/WJetsToLNu_HT-600To800_OtherPart.root"],
		"WJetsToLNu_HT-800To1200": [my_background_base + "WJetsToLNu_HT-800To1200_26August25_0708_skim_Newskim/WJetsToLNu_HT-800To1200.root", my_background_base + "WJetsToLNu_HT-800To1200_OtherPart_26August25_0925_skim_Newskim/WJetsToLNu_HT-800To1200_OtherPart.root"],
		"WJetsToLNu_HT-1200To2500": [my_background_base + "WJetsToLNu_HT-1200To2500_26August25_1016_skim_Newskim/WJetsToLNu_HT-1200To2500.root", my_background_base + "WJetsToLNu_HT-1200To2500_OtherPart_26August25_1041_skim_Newskim/WJetsToLNu_HT-1200To2500_OtherPart.root"],
		"WJetsToLNu_HT-2500ToInf": [my_background_base + "WJetsToLNu_HT-2500ToInf_26August25_1047_skim_Newskim/WJetsToLNu_HT-2500ToInf.root", my_background_base + "WJetsToLNu_HT-2500ToInf_OtherPart_26August25_1043_skim_Newskim/WJetsToLNu_HT-2500ToInf_OtherPart.root"]
	}

	Ganesh_background_dict = {
		"TTToSemiLeptonic": [Ganesh_background_base + "TTToSemiLeptonic.root",Ganesh_background_base + "TTToSemiLeptonic_2.root",Ganesh_background_base + "TTToSemiLeptonic_3.root",Ganesh_background_base + "TTToSemiLeptonic_4.root"], 
		"TTTo2L2Nu": [Ganesh_background_base + "TTTo2L2Nu.root",Ganesh_background_base + "TTTo2L2Nu_2.root"], 
		"TTToHadronic": [Ganesh_background_base + "TTToHadronic.root"],
		"ZZ4l": [Ganesh_background_base + "ZZTo4L.root"], 
		"VV2l2nu": [Ganesh_background_base + "WWTo2L2Nu.root"], 
		"WZ1l3nu": [Ganesh_background_base + "WZTo1L3Nu_4f.root"], 
		"ZZ2l2q": [Ganesh_background_base + "ZZTo2Q2L_mllmin4p0.root"],
		"WZ2l2q": [Ganesh_background_base + "WZTo2Q2L_mllmin4p0.root"],
		"WZ1l1nu2q" : [Ganesh_background_base + "WZTo1L1Nu2Q_4f.root"],
        #"DYJetsToLL_M-4to50_HT-70to100": [Ganesh_background_base + "DYJetsToLL_M-4to50_HT-70to100.root"], Can't access this info, 0 events
        #"DYJetsToLL_M-4to50_HT-100to200": [Ganesh_background_base + "DYJetsToLL_M-4to50_HT-100to200.root"],
        "DYJetsToLL_M-4to50_HT-200to400": [Ganesh_background_base + "DYJetsToLL_M-4to50_HT-200to400.root"],
        "DYJetsToLL_M-4to50_HT-400to600": [Ganesh_background_base + "DYJetsToLL_M-4to50_HT-400to600.root"],
        "DYJetsToLL_M-4to50_HT-600toInf": [Ganesh_background_base + "DYJetsToLL_M-4to50_HT-600toInf.root"],
        "DYJetsToLL_M-50_HT-70to100": [Ganesh_background_base + "DYJetsToLL_M-50_HT-70to100.root"],
        "DYJetsToLL_M-50_HT-100to200": [Ganesh_background_base + "DYJetsToLL_M-50_HT-100to200.root"],
        "DYJetsToLL_M-50_HT-200to400": [Ganesh_background_base + "DYJetsToLL_M-50_HT-200to400.root"],
        "DYJetsToLL_M-50_HT-400to600": [Ganesh_background_base + "DYJetsToLL_M-50_HT-400to600.root"],
        "DYJetsToLL_M-50_HT-600to800": [Ganesh_background_base + "DYJetsToLL_M-50_HT-600to800.root"],
        "DYJetsToLL_M-50_HT-800to1200": [Ganesh_background_base + "DYJetsToLL_M-50_HT-800to1200.root"],
        "DYJetsToLL_M-50_HT-1200to2500": [Ganesh_background_base + "DYJetsToLL_M-50_HT-1200to2500.root"],
        "DYJetsToLL_M-50_HT-2500toInf": [Ganesh_background_base + "DYJetsToLL_M-50_HT-2500toInf.root"],
		"T-tchan": [Ganesh_background_base + "ST_t-channel_top_4f_InclusiveDecays.root"], 
		"Tbar-tchan": [Ganesh_background_base + "ST_t-channel_antitop_4f_InclusiveDecays.root"], 
		"T-tW": [Ganesh_background_base + "ST_tW_top_5f_inclusiveDecays.root"], 
		"Tbar-tW": [Ganesh_background_base + "ST_tW_antitop_5f_inclusiveDecays.root"],
		"WJetsToLNu_HT-100To200": [Ganesh_background_base + "WJetsToLNu_HT-100To200.root"],
		"WJetsToLNu_HT-200To400": [Ganesh_background_base + "WJetsToLNu_HT-200To400.root"], 
		"WJetsToLNu_HT-400To600": [Ganesh_background_base + "WJetsToLNu_HT-400To600.root", Ganesh_background_base +"WJetsToLNu_HT-400To600_2.root"], 
		"WJetsToLNu_HT-600To800": [Ganesh_background_base + "WJetsToLNu_HT-600To800.root", Ganesh_background_base + "WJetsToLNu_HT-600To800_2.root"],
		"WJetsToLNu_HT-800To1200": [Ganesh_background_base + "WJetsToLNu_HT-800To1200.root", Ganesh_background_base + "WJetsToLNu_HT-800To1200_2.root"],
		"WJetsToLNu_HT-1200To2500": [Ganesh_background_base + "WJetsToLNu_HT-1200To2500.root", Ganesh_background_base + "WJetsToLNu_HT-1200To2500_2.root"],
		"WJetsToLNu_HT-2500ToInf": [Ganesh_background_base + "WJetsToLNu_HT-2500ToInf.root", Ganesh_background_base + "WJetsToLNu_HT-2500ToInf_2.root"]
	}

	#Produce cutflow csv table
	csv_file_fields = ["Sample", "My_LumiWeight","Ganesh_LumiWeight","Relative_Difference"]
	table_array = []
	for file_name in my_background_dict.keys():
		sample_dict = dict.fromkeys(csv_file_fields)
		sample_dict["Sample"] = file_name
		#Obtain my value of the lumi weight
		sum_GenWeights = 0
		for file in my_background_dict[file_name]:
			with uproot.open(file) as tempFile:
				sum_GenWeights += np.sum(tempFile['Runs/genEventSumw'].array())
		myWeight = weight_calc(file_name,sum_GenWeights)
		sample_dict["My_LumiWeight"] = myWeight

		#Obtain Ganesh's value of the lumi weight
		ganeshWeight = 0
		for file in Ganesh_background_dict[file_name]:
			with uproot.open(file) as tempFile:
				print(file_name)
				print(tempFile["Events/xsWeight"].array()[0])
				print(tempFile["Events/genWeight"].array()[0])
				ganeshWeight += tempFile["Events/xsWeight"].array()[0]/tempFile["Events/genWeight"].array()[0]
		sample_dict["Ganesh_LumiWeight"] = ganeshWeight

		#Obtain relative difference between the two
		rel_diff = (myWeight - ganeshWeight)/ganeshWeight * 100
		sample_dict["Relative_Difference"] = rel_diff
		table_array.append(sample_dict)
		#cutflow_table_array.append(fourtau_out[file]["cutflow_dict"])
	
	#Output table	
	with open("LumiWeight_Comp.csv", mode = "w", newline = '') as file:	
		writer = csv.DictWriter(file,fieldnames=csv_file_fields)
		writer.writeheader()
		writer.writerows(table_array)
	print("Table Out")
