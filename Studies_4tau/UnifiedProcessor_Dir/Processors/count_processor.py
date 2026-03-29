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

class CountingProcessor(processor.ProcessorABC):
	def __init__(self):
		pass
	def process(self,events):
		dataset = events.metadata['dataset']
		event_count = ak.num(events.boostedTau_pt,axis=0)

		return {dataset: {"n_events": event_count}}

	def postprocess(self, accumulator):
		pass


