# from ..utils import cluster_parameters
# from pexpect import pxssh

import paramiko, socket, os, pickle
from datetime import datetime

import pandas as pd

def get_session_data():

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
	username = 'schmidt124'
	proxyServerName = 'login.gwdg.de'
	serverName = 'login-dbn02.hpc.gwdg.de'

	ssh_key_file = '/home/wollex/.ssh/id_ed25519_CIDBN'
	
	client = establish_connection(serverName,username,ssh_key_file,proxyJump=proxyServerName)
	sftp_client = client.open_sftp()

	source_folder = '/usr/users/cidbn1/neurodyn/AlzheimerMice_Hayashi'
	processed_folder = '/usr/users/cidbn1/placefields/AlzheimerMice_Hayashi'

	tmp_folder = '../data/tmp'

	label_levels = ['Mouse','Session']
	index = pd.MultiIndex(levels=[[],[]],codes=[[],[]],names=label_levels)

	columns = ['files_recording','files_behavior','date','time','consistent']
	df = pd.DataFrame(columns=columns,index=index)

	# parse data from mice
	_, stdout, stderr = client.exec_command(f'ls {source_folder}')
	mice = str(stdout.read(), encoding='utf-8').splitlines()
	for m,mouse in enumerate(mice):
		print(mouse)
		mouseFolder = os.path.join(source_folder,mouse)

		file = 'matching/neuron_registration_.pkl'
		remote_file = os.path.join(mouseFolder,file)
		tmp_file = os.path.join(tmp_folder,file)

		sftp_client.get(remote_file,tmp_file)

		with open(tmp_file,'rb') as f_open:
			ld = pickle.load(f_open)
		
		# stopped here: adjust matching stuff, such that waaaaay less data is stored (only necessary stuff): remove double SNR, cnn, rvals; p_same, cm, ...
		# flow could be stored extra...
		# also: try predefining fields of dictionary 


		_, stdout, stderr = client.exec_command(f'ls {mouseFolder}')
		sessions = str(stdout.read(), encoding='utf-8').splitlines()
		for session in sessions:
			print(session)

			sessionFolder = os.path.join(mouseFolder,session)

			new_data = {
				'files_recording': False,
				'files_behavior': False,
				# 'date': None,
				# 'time': None,
				# 'consistent': False
			}

			_, stdout, stderr = client.exec_command(f'ls {sessionFolder}')
			files = str(stdout.read(), encoding='utf-8').splitlines()

			## check, whether imaging data is present
			if 'images' in files:
				# _, stdout, stderr = client.exec_command(f'ls {os.path.join(data_folder,mouse,session,"images")} | wc -l')
				# nFiles = int(stdout.read().splitlines()[0])
				# if nFiles > 2000:	## arbitrary threshold, just checking if some files are there - could also be checking for exactly 8989
				# imagesPresent = True
				new_data['files_recording'] = True

			for file in files:
				if file.startswith('aa') | file.startswith('crop'):
					new_data['files_behavior'] = True

					if file.startswith('aa'):
						## get date from name
						fileparts = file.split('_')
						date = datetime.strptime(fileparts[0][2:8],'%m%d%y').date()
						time = fileparts[2][:2]

						sameMouse = fileparts[1]==mouse

						new_data['date'] = date
						new_data['time'] = time
						new_data['consistent'] = sameMouse
			

			## if data can be processed, check for further details
			if new_data['files_behavior'] & new_data['files_recording']:
				
				## obtain 
				## 		behavior data: reward location (+ gate location, reward probability, delay)
				## 		alignment data: shift and correlation
				## 		neuron data: number of neurons & place cells
				
				sessionFolder_processed = os.path.join(processed_folder,mouse,session)

				file = 'aligned_behavior.pkl'
				remote_file = os.path.join(sessionFolder_processed,file)
				tmp_file = os.path.join(tmp_folder,file)


				sftp_client.get(remote_file,tmp_file)

				with open(tmp_file,'rb') as f_open:
					aligned_behavior = pickle.load(f_open)
					new_data['reward_location'] = aligned_behavior['reward_location']
					new_data['reward_prob'] = aligned_behavior['reward_prob']
				os.remove(tmp_file)

				# read matching/neuron_registration.pkl to get number of neurons and alignment statistics
				# read aligned_behavior.pkl to get reward location (and maybe reward_probability)

				

				
				# print(file)
				# if 'CaImAn_complete.hdf5' in file:
				# 	print('found')
				# 	# break

			new_idx = pd.MultiIndex.from_tuples([(mouse,session)],names=label_levels)
			new_df = pd.DataFrame(new_data,index=new_idx)

			df = pd.concat([df,new_df])

			# break
		# if m>2:
		break
	# stdin, stdout, stderr = client.exec_command('ls')
	# print(stderr.read())
	client.close()
	return df
	
 

def establish_connection(serverName,username,ssh_key_file,proxyJump=None):

	## getting IP address of server, as hostname can not be directly used
	serverIP = socket.getaddrinfo(serverName, None)[0][4][0]	## kinda hacky, but works...
	pkey = paramiko.Ed25519Key.from_private_key_file(ssh_key_file)

	client = paramiko.SSHClient()
	client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

	if proxyJump:

		client_proxy = establish_connection(proxyJump,username,ssh_key_file)

		transport = client_proxy.get_transport()
		dest_addr = (serverIP, 22)		
		local_addr = ('127.0.0.1', 22)	## does this always hold for current host?
		channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)
		client.connect(serverIP, username=username, pkey=pkey, sock=channel)
	else:
		client.connect(serverIP, username=username, pkey=pkey)


	return client
