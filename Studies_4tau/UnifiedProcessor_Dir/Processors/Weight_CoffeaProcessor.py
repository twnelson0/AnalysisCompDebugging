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

class CountingProcessor(processor.ProcessorABC):
	def __init__(self):
		self.isData = False
		#pass
	def process(self,events):
		dataset = events.metadata['dataset']
		if ("Data_" in dataset):
			self.isData = True

		if (self.isData):
			#event_count = np.sum(events.genEventCount)
			event_count = 1 
			genWeightSum = 1 
		else:
			event_count = np.sum(events.genEventCount)
			genWeightSum = np.sum(events.genEventSumw)

		return {dataset: {
            "n_events": event_count,
            "genWeightSum": genWeightSum
            }
        }

	def postprocess(self, accumulator):
		pass

