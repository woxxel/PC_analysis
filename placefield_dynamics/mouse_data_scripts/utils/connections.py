import paramiko, socket


def establish_connection(serverName, username, ssh_key_file, proxyJump=None):

    ## getting IP address of server, as hostname can not be directly used
    serverIP = socket.getaddrinfo(serverName, None)[0][4][
        0
    ]  ## kinda hacky, but works...
    pkey = paramiko.Ed25519Key.from_private_key_file(ssh_key_file)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    if proxyJump:

        client_proxy = establish_connection(proxyJump, username, ssh_key_file)

        transport = client_proxy.get_transport()
        dest_addr = (serverIP, 22)
        local_addr = ("127.0.0.1", 22)  ## does this always hold for current host?
        channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)
        client.connect(serverIP, username=username, pkey=pkey, sock=channel)
    else:
        client.connect(serverIP, username=username, pkey=pkey)

    return client


def set_hpc_params(hpc="sofja"):

    ## setting up connection to server
    ssh_key_file = f"/home/wollex/.ssh/id_ed25519_GWDG"
    username = "schmidt124"
    if hpc == "sofja":
        proxyServerName = "glogin.hpc.gwdg.de"
        serverName = "login-dbn02.hpc.gwdg.de"
        client = establish_connection(
            serverName, username, ssh_key_file, proxyJump=proxyServerName
        )
    else:
        proxyServerName = "glogin.gwdg.de"
        serverName = "gwdu101.gwdg.de"
        client = establish_connection(
            serverName, username, ssh_key_file, proxyJump=proxyServerName
        )

    path_code = "~/program_code/PC_analysis"

    ## setting sbatch parameters
    batch_params = {}
    batch_params["submit_file"] = f"{path_code}/sbatch_submit.sh"

    if hpc == "sofja":
        batch_params["A"] = "cidbn_legacy"
        batch_params["p"] = "cidbn"
    else:
        batch_params["A"] = "scc_users"
        batch_params["p"] = "medium"

    return client, path_code, batch_params
