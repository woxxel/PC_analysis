import os


def copy_to_server(path_source, path_target, ssh_conn, show_progress=True):
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



    ## assume path source contains mouse data folders, which should be copied entirely

    permissions = "-p --chmod=Du=rwx,Dg=rwx,Do=,Fu=rw,Fg=rw,Fo="

    try:
        for dir_mouse in os.listdir(path_source):
            path_mouse = os.path.join(path_source, '.', dir_mouse)

            if os.path.isdir(path_mouse):
                
                path_names = os.listdir(path_mouse)
                path_names.sort()
                for path_name in path_names:
                    
                    ## get Dates file, if present
                    if path_name.startswith('Dates'):

                        cp_cmd = f"rsync --relative {permissions} -e ssh {os.path.join(path_mouse,path_name)} {ssh_conn}:{path_target}"
                        os.system(cp_cmd)
                    
                    
                    path_session = os.path.join(path_mouse, path_name)
                    if os.path.isdir(path_session) and path_name.startswith('Session'):


                        print(f"mouse {dir_mouse}, {path_name}")
                        
                        for file_name in os.listdir(path_session):
                            ## get behavior file
                            if file_name.endswith(".txt"):  #file_name.startswith("aa") and 
                                cp_cmd = f"rsync --relative {permissions} -e ssh {os.path.join(path_session,file_name)} {ssh_conn}:{path_target}"
                                
                                # print(f"Pushing behavior data to {ssh_conn}:{os.path.join(path_target,dir_mouse,path_name)}...")
                                # print(cp_cmd)
                                os.system(cp_cmd)

                            ## get imaging data
                            if file_name.endswith('.h5'):
                                cp_cmd = f"rsync -z --relative {permissions} -e ssh {os.path.join(path_session,file_name)} {ssh_conn}:{path_target}"
                                
                                print(f"Pushing imaging data to {ssh_conn}:{os.path.join(path_target,dir_mouse,path_name)}...")
                                os.system(cp_cmd)
                    # break
            # break
    except KeyboardInterrupt:
        print("Interrupted!")
        return