import paramiko, socket

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