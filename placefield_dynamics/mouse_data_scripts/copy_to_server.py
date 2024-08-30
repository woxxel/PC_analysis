import os

from .utils.connections import *

def copy_to_server(path_source, path_target,ssh_conn='hpc-sofja',ssh_config_file_name='id_ed25519_GWDG'):
    """
        Pushes data to the GWDG servers using rsync

        input:
            path_source (int)
                path to the session folder to be accessed
            path_target (string)
                path to a folder in which to store the pushed data
            ssh_alias (string)
                an alias for an ssh connection to the GWDG server - make sure to have it properly set up with an account with appropriate rights to access the given filestructures

        returns:
            nothing
    """
    ## setting up connection to server
    username = 'schmidt124'
    # proxyServerName = 'login.gwdg.de'
    proxyServerName = 'login.gwdg.de'
    serverName = 'login-dbn02.hpc.gwdg.de'

    ssh_key_file = f'/home/wollex/.ssh/{ssh_config_file_name}'

    client = establish_connection(serverName,username,ssh_key_file,proxyJump=proxyServerName)
    sftp_client = client.open_sftp()

    ## assume path source contains mouse data folders, which should be copied entirely
    for dir_mouse in os.listdir(path_source):
        path_mouse = os.path.join(path_source, '.', dir_mouse)

        if os.path.isdir(path_mouse):
            
            path_names = os.listdir(path_mouse)
            path_names.sort()
            for path_name in path_names:
                
                ## get Dates file, if present
                if path_name.startswith('Dates'):
                    
                    copy_to_remote(sftp_client,path_source,path_target,os.path.join(dir_mouse,path_name),ssh_conn)
                
                
                path_session = os.path.join(path_mouse, path_name)
                if os.path.isdir(path_session) and path_name.startswith('Session'):
                    
                    # if path_name=='Session47' and dir_mouse=='232':
                    #     continue

                    print(f"mouse {dir_mouse}, {path_name}")
                    
                    for file_name in os.listdir(path_session):
                        ## get behavior file
                        if file_name.endswith(".txt") or file_name.endswith('.h5'):  
                            copy_to_remote(sftp_client,path_source,path_target,os.path.join(dir_mouse,path_name,file_name),ssh_conn,printing=True)

                # break
        # break
    # except KeyboardInterrupt:
    #     print("Interrupted!")
    #     return
    

def copy_to_remote(sftp_client,path_source,path_target,relative_path_to_file, ssh_conn, permissions="-p --chmod=Du=rwx,Dg=rwx,Do=,Fu=rw,Fg=rw,Fo=", printing=False):
    
    infile = os.path.join(path_source,'.',relative_path_to_file)
    outfile = os.path.join(path_target,relative_path_to_file)
    try:
        sftp_client.stat(outfile)
        print(f'{outfile} already exists - skipping')
    except IOError:
        cp_cmd = f"rsync --info=progress2 --relative {permissions} -e ssh {infile} {ssh_conn}:{path_target}"
        if printing:
            print(f"Pushing behavior data to {ssh_conn}:{outfile}...")
        os.system(cp_cmd)
