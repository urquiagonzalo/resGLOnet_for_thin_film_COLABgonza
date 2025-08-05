import numpy as np
import pandas as pd
import torch

class MatDatabase(object):
	"""docstring for MatDatabase
		Parameters: 
			material_key: list of material names
	"""
	def __init__(self, material_key):
		super(MatDatabase, self).__init__()
		self.material_key = material_key
		self.num_materials = len(material_key)
		self.mat_database = self.build_database()

	def build_database(self):
		mat_database = {}
		
		#%% Read in the dispersion data of each material
		for i in range(self.num_materials):
			file_name = './material_database/mat_' + self.material_key[i] + '.xlsx'
			
			try: 
				A = np.array(pd.read_excel(file_name))
				mat_database[self.material_key[i]] = (A[:, 0], A[:, 1], A[:, 2])
			except NameError:
				print('The material database does not contain', self.material_key[i])

		return mat_database


	def interp_wv(self, wv_in, material_key, ignoreloss = False):
		'''
			parameters
				wv_in (tensor) : number of wavelengths
				material_key (list) : number of materials

			return
				refractive indices (tensor or tuple of tensor) : number of materials x number of wavelengths
		'''
		n_data = np.zeros((len(material_key), wv_in.size(0)), dtype=np.float32)
		k_data = np.zeros((len(material_key), wv_in.size(0)), dtype=np.float32)
		for i in range(len(material_key)):
			mat = self.mat_database[material_key[i]]
			n_data[i, :] = np.interp(wv_in, mat[0], mat[1])
			k_data[i, :] = np.interp(wv_in, mat[0], mat[2])

		if ignoreloss:
			#"corrección GMU: torch.zeros_like(...) espera un tensor de PyTorch como entrada, pero le estámos pasando un numpy.ndarray.
			#Corregimos la siguiente línea
			#return torch.complex(torch.tensor(n_data), torch.zeros_like(k_data))
			k_data_tensor = torch.tensor(k_data)
			return torch.complex(torch.tensor(n_data), torch.zeros_like(k_data_tensor))
		else:
			return torch.complex(torch.tensor(n_data), torch.tensor(k_data))
		
