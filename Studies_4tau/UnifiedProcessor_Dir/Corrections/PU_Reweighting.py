#Code based on: https://gitlab.cern.ch/ckoraka/vll-analysis/-/blob/master/vll/scalefactors/puScalefactors.py 

import os.path
import correctionlib 

from coffea import util
from coffea import lookup_tools
#from coffea.lookup_tools import extractor

import numpy as np
import uproot
import awkward as ak

###### Pileup reweighing
##############################################
## Get central PU data and MC profiles and calculate reweighting
## Using the current UL recommendations in:
##   https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData
##   - 2018: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/
##   - 2017: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/
##   - 2016: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/
##
## MC histograms from:
##    https://github.com/CMS-LUMI-POG/PileupTools/

#PP cross sections for PU
pu_w = 69200
pu_w_up = 72400
pu_w_down = 66000

#Get current working directory
cwd = os.path.dirname(__file__)
pu_path = f'{cwd}/pileup/'

MCPUfile = {'2016pre':pu_path + 'pileup_2016BF.root', '2016post':pu_path + 'pileup_2016GH.root', 
		'2017':pu_path + 'pileup_2017_shifts.root', '2018':pu_path + 'pileup_2018_shifts.root'}

#MCPUfile = {'2016pre':'Corrections/pileup/pileup_2016BF.root', '2016post':'Corrections/pileup/pileup_2016GH.root', 
#		'2017':'Corrections/pileup/pileup_2017_shifts.root', '2018':'Corrections/pileup/pileup_2018_shifts.root'}

#File name wiht pu observed distribution 
def GetDataPUname(era, var = ""):
	if era == '2016pre': era = '2016-preVFP'
	if era == '2016post': era = '2016-postVFP'
	if var == 'nominal': ppxsec = pu_w
	elif var == 'up': ppxsec = pu_w_up
	elif var == 'down': ppxsec = pu_w_down

	return 'PileupHistogram-goldenJSON-13tev-%s-%sub-99bins.root'%((era), str(ppxsec))

PUfunc = {}
era_origin_dict = {'2016pre': '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/',
	'2016post': '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/',
	'2017': '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/',
	#'2018': '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/'}
	'2018': pu_path}

### Load histograms and get lookup tables (extractors are not working here...)
for era in ['2016pre','2016post','2017','2018']:
  PUfunc[era] = {}
  with uproot.open(MCPUfile[era]) as fMC:
    hMC = fMC['pileup']
    PUfunc[era]['MC'] = lookup_tools.dense_lookup.dense_lookup(hMC.values() / np.sum( hMC.values()), hMC.axis(0).edges())
  with uproot.open(era_origin_dict[era]+GetDataPUname(era,  'nominal')) as fData:
    hD = fData['pileup']
    PUfunc[era]['Data'] = lookup_tools.dense_lookup.dense_lookup(hD.values() / np.sum(hD.values()), hD.axis(0).edges())
  with uproot.open(era_origin_dict[era]+GetDataPUname(era,  'up')) as fDataUp:
    hDUp = fDataUp['pileup']
    PUfunc[era]['DataUp'] = lookup_tools.dense_lookup.dense_lookup(hDUp.values() / np.sum(hDUp.values()), hD.axis(0).edges())
  with uproot.open(era_origin_dict[era]+GetDataPUname(era, 'down')) as fDataDo:
    hDDo = fDataDo['pileup']
    PUfunc[era]['DataDo'] = lookup_tools.dense_lookup.dense_lookup(hDDo.values() / np.sum(hDDo.values()), hD.axis(0).edges())


def getPUSF(nTrueInt, era, var = 'nominal'):
	era_array = ['2016pre','2016post','2017','2018']
	
	if (era not in era_array): raise Exception(f"Error: Unknown era \"{era}\"")
	if (var not in ["nominal","up","down"]): raise Exception(f"Error: Unkown var \"{var}\"")

	nMC  =PUfunc[era]['MC'](nTrueInt+1)
	nData=PUfunc[era]['DataUp' if var == 'up' else ('DataDo' if var == 'down' else 'Data')](nTrueInt)
	weights = np.divide(nData,nMC)
	return weights

