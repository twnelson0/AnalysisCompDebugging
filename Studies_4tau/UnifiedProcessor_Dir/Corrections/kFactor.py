import hist
import uproot
import numpy as np
from coffea.lookup_tools import extractor

ext = extractor()

#Import kfactor histograms
ext.add_weight_sets(["nlo_EW_WJets EWK_W_nominal Corrections/ZorW_NLO_corrections.root"])
ext.add_weight_sets(["nlo_EW_ZJets EWK_Z_nominal Corrections/ZorW_NLO_corrections.root"])
ext.add_weight_sets(["nlo_QCD_WJets QCD_W Corrections/ZorW_NLO_corrections.root"])
ext.add_weight_sets(["nlo_QCD_ZJets QCD_Z Corrections/ZorW_NLO_corrections.root"])

ext.finalize()
evaluator = ext.make_evaluator()

#k Factor Functions
#Old inccorrect evaluators
#def getEWKW(pt):
#	return evaluator["nlo_EW_WJets"](pt)
#
#def getEWKZ(pt):
#	return evaluator["nlo_EW_ZJets"](pt)

#Single particle electroweak corrections
def getEWKW_Ind(pt):
	weight = 1
	if 0 <= pt < 170:
		weight = 0.94269
	elif 170 <= pt < 200:
		weight = 0.902615
	elif 200 <= pt < 230:
		weight = 0.898827
	elif 230 <= pt < 260:
		weight = 0.959081
	elif 260 <= pt < 290:
		weight = 0.891248
	elif 290 <= pt < 320:
		weight = 0.860188
	elif 320 <= pt < 350:
		weight = 0.884811
	elif 350 <= pt < 390:
		weight = 0.868131
	elif 390 <= pt < 430:
		weight = 0.848655
	elif 430 <= pt < 470:
		weight = 0.806186
	elif 470 <= pt < 510:
		weight = 0.848507
	elif 510 <= pt < 550:
		weight = 0.83763
	elif 550 <= pt < 590:
		weight = 0.792152
	elif 590 <= pt < 640:
		weight = 0.730731
	elif 640 <= pt < 690:
		weight = 0.778061
	elif 690 <= pt < 740:
		weight = 0.771811
	elif 740 <= pt < 790:
		weight = 0.795004
	elif 790 <= pt < 840:
		weight = 0.757859
	elif 840 <= pt < 900:
		weight = 0.709571
	elif 900 <= pt < 960:
		weight = 0.702751
	elif 960 <= pt < 1020:
		weight = 0.657821
	elif 1020 <= pt < 1090:
		weight = 0.762559
	elif 1090 <= pt < 1160:
		weight = 0.845925
	elif pt >= 1160:
		weight = 0.674034
	
	return weight

def getEWKZ_Ind(pt):
	weight = 1
	if 0 <= pt < 170:
		weight = 1.43896
	elif 170 <= pt < 200:
		weight = 1.45307
	elif 200 <= pt < 230:
		weight = 1.41551
	elif 230 <= pt < 260:
		weight = 1.42199
	elif 260 <= pt < 290:
		weight = 1.3477
	elif 290 <= pt < 320:
		weight = 1.35302
	elif 320 <= pt < 350:
		weight = 1.34289
	elif 350 <= pt < 390:
		weight = 1.32474
	elif 390 <= pt < 430:
		weight = 1.23267
	elif 430 <= pt < 470:
		weight = 1.22641
	elif 470 <= pt < 510:
		weight = 1.23149
	elif 510 <= pt < 550:
		weight = 1.21593
	elif 550 <= pt < 590:
		weight = 1.16506
	elif 590 <= pt < 640:
		weight = 1.01718
	elif 640 <= pt < 690:
		weight = 1.01575
	elif 690 <= pt < 740:
		weight = 1.05425
	elif 740 <= pt < 790:
		weight = 1.05992
	elif 790 <= pt < 840:
		weight = 1.01503
	elif 840 <= pt < 900:
		weight = 1.01761
	elif 900 <= pt < 960:
		weight = 0.947194
	elif 960 <= pt < 1020:
		weight = 0.932754
	elif 1020 <= pt < 1090:
		weight = 1.00849
	elif 1090 <= pt < 1160:
		weight = 0.94805
	elif pt >= 1160:
		weight = 0.86956
	
	return weight

#Vectorized EWK corrections
getEWKW = np.vectorize(getEWKW_Ind)
getEWKZ = np.vectorize(getEWKZ_Ind)

#QCD Correctiosn
def getQCDW(pt):
	return evaluator["nlo_QCD_WJets"](pt)

def getQCDZ(pt):
	return evaluator["nlo_QCD_ZJets"](pt)
