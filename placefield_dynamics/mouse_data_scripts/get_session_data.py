# from ..utils import cluster_parameters
# from pexpect import pxssh

import os, pickle, tqdm
from itertools import combinations
from matplotlib import pyplot as plt

from caiman.source_extraction.cnmf.cnmf import load_dict_from_hdf5
from ..neuron_matching.utils import calculate_img_correlation
from datetime import datetime
import numpy as np

import pandas as pd
from align_mouse_behavior import *

from .utils.connections import *

def get_session_data(ssh_config_file_name='id_ed25519_GWDG',datasets=['Shank2Mice_Hayashi','AlzheimerMice_Hayashi'],suffix='complete'):

	'''
		Design choices:

			* classification works only if matching has been performed?

		function to check for presence and logic of data

		1. check if files of
			recording (?) 
			neuron detection
			mouse behavior
		are present

		2. check if neuron matching is present 
			and use to check whether some data seems flawed
			(e.g. large shifts, twice the same measurement, bad detection results, ...)

		3. check if data is consistent with the mouse

		stores:
			* paths to sessions and data
			* whether all necessary files are present
			* reward location


	'''

	## setting up connection to server
	username = 'schmidt124'
	proxyServerName = 'login.gwdg.de'
	serverName = 'login-dbn02.hpc.gwdg.de'

	ssh_key_file = f'/home/wollex/.ssh/{ssh_config_file_name}'
	
	client = establish_connection(serverName,username,ssh_key_file,proxyJump=proxyServerName)
	sftp_client = client.open_sftp()


	## setting paths on server
	if suffix and not suffix.startswith('_'):
		suffix = '_'+suffix

	source_folder = '/usr/users/cidbn1/neurodyn'#/AlzheimerMice_Hayashi'
	processed_folder = '/usr/users/cidbn1/placefields'#/AlzheimerMice_Hayashi'


	## setting temporary path on local machine for storing data
	tmp_folder = '../data/tmp'


	## prepare data structure
	columns = ['files_recording','files_processed_recording','recording_names','mouse_from_recording','time_from_recording','files_behavior','files_processed_behavior','mouse_from_behavior','date_from_behavior','time_from_behavior','consistent','reward_location','reward_prob','x_shift','y_shift','image_correlation','duplicate','n_neurons']
	
	label_levels = ['Mouse','Session']
	index = pd.MultiIndex(levels=[[],[]],codes=[[],[]],names=label_levels)

	## prepare looping over datasets
	# _, stdout, stderr = client.exec_command(f'ls {source_folder}')
	# datasets = str(stdout.read(), encoding='utf-8').splitlines()
	# datasets = ['Shank2Mice_Hayashi']
	# datasets = ['Shank2Mice_Hayashi','AlzheimerMice_Hayashi']

	for d,dataset in enumerate(datasets):
		
		df = pd.DataFrame(columns=columns,index=index)
		df_missing = pd.DataFrame(columns=columns,index=index)

		# parse data from mice
		dataset_processed_folder = os.path.join(processed_folder,dataset)
		dataset_source_folder = os.path.join(source_folder,dataset)
		_, stdout, stderr = client.exec_command(f'ls {dataset_source_folder}')
		mice = str(stdout.read(), encoding='utf-8').splitlines()
		for m,mouse in enumerate(mice):
			# if m==0: continue
			print('mouse: ',mouse)
			if not mouse[:2].isdigit():
				continue
			mouseFolder = os.path.join(dataset_source_folder,mouse)

			file = f'matching/neuron_registration{suffix}.pkl'
			remote_file = os.path.join(dataset_processed_folder,mouse,file).strip()
			tmp_file = os.path.join(tmp_folder,file).strip()

			# print(os.path.dirname(tmp_file))
			if not os.path.exists(os.path.dirname(tmp_file)):
				os.makedirs(os.path.dirname(tmp_file))

			# print(remote_file,tmp_file)
			# read matching/neuron_registration.pkl to get number of neurons and alignment statistics
			try:
				sftp_client.get(remote_file,tmp_file)
			except:
				print('no matching file found')
				continue

			with open(tmp_file,'rb') as f_open:
				ld = pickle.load(f_open)

			# print('loaded matching file with keys: ',ld.keys())
			shifts = ld['remap']['shift']
			correlations = ld['remap']['corr']

			n_neurons = np.isfinite(ld['assignments']).sum(axis=0)


			# sessions = ld['filePath']
			# print(sessions)
			# stopped here: adjust matching stuff, such that waaaaay less data is stored (only necessary stuff): remove double SNR, cnn, rvals; p_same, cm, ...
			# flow could be stored extra...
			# also: try predefining fields of structure (such that no __dict__ is created)

			_, stdout, stderr = client.exec_command(f'ls {mouseFolder} | grep Session')
			sessions = str(stdout.read(), encoding='utf-8').splitlines()
			iterate_sessions = tqdm.tqdm(enumerate(sessions))
			for s,session in iterate_sessions:

				# session = os.path.dirname(session).split('/')[-1]
				iterate_sessions.set_description(f'now session: {session}')
				new_data = {
					'files_recording': False,
					'files_processed_recording': False,
					'files_behavior': False,
					'files_processed_behavior': False,
					'consistent': True
				}

				if session:
					new_data['x_shift'] = shifts[s,0]
					new_data['y_shift'] = shifts[s,1]
					new_data['image_correlation'] = correlations[s]

					max_idx = np.where(ld['Cn_corr'][s,:]>0.9)[0]
					new_data['duplicate'] = sessions[max_idx[0]] if len(max_idx) else None
					
					new_data['n_neurons'] = n_neurons[s]
					sessionFolder = os.path.join(mouseFolder,session)


					_, stdout, stderr = client.exec_command(f'ls {sessionFolder}')
					files = str(stdout.read(), encoding='utf-8').splitlines()

					## check, whether imaging data is present
					if 'images' in files:
						# _, stdout, stderr = client.exec_command(f'ls {os.path.join(data_folder,mouse,session,"images")} | wc -l')
						# nFiles = int(stdout.read().splitlines()[0])
						# if nFiles > 2000:	## arbitrary threshold, just checking if some files are there - could also be checking for exactly 8989
						# imagesPresent = True
						new_data['files_recording'] = True

						imageFolder = os.path.join(sessionFolder,'images')
						_, stdout, stderr = client.exec_command(f'ls -f1 {imageFolder} | head -5')
						images = str(stdout.read(), encoding='utf-8').splitlines()
						for image in images:
							if image.endswith('.tif'):
								## get mouse name and recording time from name (hope it's all homogenoeusly named!)
								new_data['recording_names'] = image[:-8]
								fileparts = image.split('_')
								# print(image,fileparts)
								new_data['mouse_from_recording'] = image.split('#')[-1].split('_')[0]
								new_data['consistent'] &= new_data['mouse_from_recording']==mouse[:len(new_data['mouse_from_recording'])]
								
								new_data['time_from_recording'] = fileparts[-1][:2]
								# print(new_data['mouse_from_recording'],new_data['time_from_recording'])
								break


					for file in files:
						if file.startswith('aa') | file.startswith('crop'):
							new_data['files_behavior'] = True

							if file.startswith('aa'):
								## get date from name
								fileparts = file.split('_')
								date = datetime.strptime(fileparts[0][2:8],'%m%d%y').date()
								time = fileparts[2][:2]

								new_data['mouse_from_behavior'] = fileparts[1]
								new_data['consistent'] &= new_data['mouse_from_behavior']==mouse

								new_data['date_from_behavior'] = date

								new_data['time_from_behavior'] = time
								if 'time_from_recording' in new_data:
									new_data['consistent'] &= new_data['time_from_behavior']==new_data['time_from_recording']
								

					## if data can be processed, check for further details
					if new_data['files_behavior'] & new_data['files_recording']:
						
						## obtain 
						## 		behavior data: reward location (+ gate location, reward probability, delay)
						## 		alignment data: shift and correlation
						## 		neuron data: number of neurons & place cells
						
						sessionFolder_processed = os.path.join(dataset_processed_folder,mouse,session)

						## find processed recording file and extract information
						file = 'OnACID_results.hdf5'
						remote_file = os.path.join(sessionFolder_processed,file)

						_, stdout, stderr = client.exec_command('test -e {0} && echo exists'.format(remote_file))
						errs = stderr.read()
						if errs:
							raise Exception('Failed to check existence of {0}: {1}'.format(remote_file, errs))
						
						new_data['files_processed_recording'] = stdout.read().strip().decode('ascii') == 'exists'


						## find processed behavior file and extract information
						# read aligned_behavior.pkl to get reward location (and maybe reward_probability)
						file = 'aligned_behavior.pkl'
						remote_file = os.path.join(sessionFolder_processed,file)
						tmp_file = os.path.join(tmp_folder,file)
						ran_alignment = False
						while True:
							try:
							# if ran_alignment:
								sftp_client.get(remote_file,tmp_file)

								with open(tmp_file,'rb') as f_open:
									aligned_behavior = pickle.load(f_open)
									new_data['reward_location'] = aligned_behavior['reward_location']
									new_data['reward_prob'] = aligned_behavior['reward_prob']
								os.remove(tmp_file)
								new_data['files_processed_behavior'] = True
								break
							except:
							# else:
								if ran_alignment:
									print('alignment failed - skipping')
									break
								
								print('no aligned_behavior.pkl found - attempting to align data')
								try:
									align_data_on_hpc(source_folder,processed_folder,dataset,mouse,session,'hpc-sofja',min_stretch=0.7)
								except:
									pass
								ran_alignment = True



					# new_idx = pd.MultiIndex.from_tuples([(mouse,session)],names=label_levels)
					# print(new_idx,new_data)
					# new_df = pd.DataFrame(new_data,index=new_idx)
					# df = pd.concat([df,new_df])
				df.loc[(mouse,session),:] = new_data
				if (not new_data['files_recording']) | (not new_data['files_behavior']) | \
					(not new_data['files_processed_recording']) | (not new_data['files_processed_behavior']) | \
					(not (new_data['duplicate'] is None)):
					# print('missing data for session: ',new_data)
					df_missing.loc[(mouse,session),:] = new_data
				
			# if m>2:
			# 	break
				
		for key in ['files_recording','files_behavior','files_processed_behavior','consistent']:
			df[key] = df[key].astype('int')
		
		write_data_to_xlsx(df,f'../data/{dataset}_data.xlsx')

		with pd.ExcelWriter(f'../data/{dataset}_missing_data.xlsx') as writer:
			# for mouse in df.index.unique('Mouse'):
			df_missing.to_excel(writer,sheet_name='mouse_missing_data')
		# write_data_to_xlsx(df_missing,f'{dataset}_missing_data.xlsx')
		



	# stdin, stdout, stderr = client.exec_command('ls')
	# print(stderr.read())
	client.close()

	return df

def write_data_to_xlsx(df,filename='session_data.xlsx'):

	with pd.ExcelWriter(filename) as writer:
		for mouse in df.index.unique('Mouse'):
			df.loc[mouse].to_excel(writer,sheet_name=mouse)
	
 



def get_video_correlation(nSes=60):

	## load background data from all sessions
	Cn = np.zeros((512,512,nSes))
	for i in range(nSes):
		try:
			Cn[...,i] = load_dict_from_hdf5(f'../data/556wt/Session{i+1:02}/CaImAn_complete.hdf5')
		except:
			pass

	## calculate correlation between all sessions
	corr = np.zeros((nSes,nSes))
	for combo in combinations(range(nSes),2):
		if Cn[...,i].sum()>0 and Cn[...,j].sum()>0:
			corr[i,j] = calculate_img_correlation(Cn[...,i],Cn[...,j])[0]
		else:
			corr[i,j] = np.NaN
		corr[j,i] = corr[i,j]
	
	plt.figure()
	plt.imshow(corr)
	plt.show(block=False)