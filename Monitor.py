import sys, time, os
import numpy as np
import lmfit as lm
from lmfit.models import PseudoVoigtModel, LinearModel
import pandas as pd
import matplotlib.pyplot as plt

# --- Monitor - Reading a '.txt' file for new data --- #
class FolderMonitor:
	def __init__(self, folder_path, fit_interval=None):
		self.folder_path = folder_path
		self.fit_interval = fit_interval
		self.processed_files = set()  # Tracking the processed files
		self.data_frame = pd.DataFrame(columns = ['theta', 'intensity', 'temp', 'max', 'file_index'])
		self.fit_data = pd.DataFrame(columns=['dois_theta_0', 'fwhm', 'area', 'temp', 'file_index', 'R-squared'])

	def run(self):
		print(f'Monitoring folder: {self.folder_path}')
		print('Waiting for XRD data! Please, wait')
		while True:
			# Current files set
			current_files = os.listdir(self.folder_path)
			# Check for new files
			new_files = set(current_files) - self.processed_files 
			data_to_read = self.order_dir(list(new_files))
			# Process new files
			for element in data_to_read:
				if element[1].endswith(".csv"):
					path = os.path.join(self.folder_path, element[1])
					data = data_read(path)
					file_index = counter()
					new_data = pd.DataFrame({'theta': [data[0]], 'intensity': [data[1]], 'temp': [data[2]], 'max': [data[1].max()], 'file_index': [file_index]})
					self.data_frame = pd.concat([self.data_frame, new_data], ignore_index=True)
					print(f"New data created at: {self.folder_path}. File name: {element[1]}")
					#print(len(self.data_frame['theta']))
					if self.fit_interval:
						fit = peak_fit(data[0], data[1], self.fit_interval)
						new_fit_data = pd.DataFrame({'dois_theta_0': [fit[0]], 'fwhm': [fit[1]], 'area': [fit[2]], 'temp': [data[2]], 'file_index': [file_index], 'R-squared': [fit[3]]})
						self.fit_data = pd.concat([self.fit_data, new_fit_data], ignore_index=True)



			# Update the set of processed files
			self.processed_files.update(current_files)  

			time.sleep(0.1) # Check for new files every second

	def order_dir(self, list):
		files_order = []
		for file in list:
			if file.endswith('.csv'):
				file_index = int(file.split(sep='_')[len(file.split(sep='_'))-2])
				files_order.append((file_index, file))

		files_order.sort()
		return files_order
	
	def set_fit_interval(self, interval):
		self.fit_interval = interval

# --- Defining the functions for data reading and peak fitting --- #
def data_read(path):
	done = False
	while not done:
		time.sleep(0.1)
		try:
			dados = pd.read_csv(path, sep=',')
			x = np.array(dados.get('2theta (degree)'))
			y = np.array(dados.get('Intensity'))
			file_name = path.split(sep='/')[len(path.split(sep='/'))-1]
			temp = None
			for i in file_name.split(sep='_'):
				if 'Celsius' in i: 
					temp = float(i.split(sep='Celsius')[0]) #Getting the temperature
			
			done = True
			return x,y,temp
		except pd.errors.EmptyDataError:
			print(f"Warning: Empty file encountered: {path}. Trying to read the data again!")
			#return None
		except Exception as e:
			print(f"An error occurred while reading data: {e}. Trying to read the data again!")
			#return None

# --- Defining the storaging lists --- #		


def peak_fit(theta, intensity, interval):
	done = False
	while not done:
		time.sleep(0.1)
		try:
			theta_fit = []
			intensity_fit = []
  
  # Slicing the data for the selected peak fitting interval #
			for i in range(len(theta)):
				if theta[i] > interval[0] and theta[i] < interval[1]: 
					theta_fit.append(theta[i])
					intensity_fit.append(intensity[i])
			theta_fit=np.array(theta_fit)
			intensity_fit=np.array(intensity_fit)
  # Building the Voigt model with lmfit #
			mod = PseudoVoigtModel(nan_policy='omit')
			pars = mod.guess(data = intensity_fit, x=theta_fit) # Initial guess to inicialize the parameters #
			#pars['gamma'].set(value=0.7, vary=True, expr='')
			bkg = LinearModel(prefix='bkg_')
			pars.update(bkg.guess(theta_fit, intensity_fit))
			mod = mod + bkg
  
			out = mod.fit(intensity_fit, pars, x=theta_fit) # Fitting the data to the Voigt model #
			comps = out.eval_components(x=theta_fit)
  # Getting the parameters from the optimal fit #
			dois_theta_0 = out.params['center']*1
			fwhm = out.params['fwhm']*1
			area = out.params['amplitude']*1
			r_squared = out.rsquared

			done = True
			return dois_theta_0, fwhm, area, r_squared
		except ValueError or TypeError as e:
			print(f'Fitting error, please wait: {e}! Please select a new fitting interval')
			done = True
			pass


# --- A counter function to index the created curves --- #
def counter():
	counter.count += 1
	return counter.count
	
counter.count = 0
monitor = FolderMonitor('/home/joao/IGAPÉ/TESTFOLDER/TEST')
#monitor.run()