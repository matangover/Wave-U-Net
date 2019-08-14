import paramiko
from getpass import getpass
import os
import sys

user = 'username'
host = 'graham.computecanada.ca'
output_dir = 'checkpoints'

def connect():
    client = paramiko.client.SSHClient()
    client.load_system_host_keys()
    password = getpass(f'Enter password for {user} on {host}: ')
    client.connect(host, username=user, password=password)
    return client

def download_checkpoint(
        ssh_client: paramiko.client.SSHClient,
        sftp_client: paramiko.sftp_client.SFTPClient,
        run_id: str
) -> None:
    remote_checkpoint_dir = f'/home/{user}/proj/Wave-U-Net/checkpoints/{run_id}/'
    _stdin, stdout, _stderr = ssh_client.exec_command(
        f'ls -1 -v {remote_checkpoint_dir} -I checkpoint | tail -n 1'
    )
    last_checkpoint_filename = stdout.read().decode('utf8')
    if not last_checkpoint_filename:
        print('File not found')
        return
    last_checkpoint_name = os.path.splitext(last_checkpoint_filename)[0]
    checkpoint_files = [
        last_checkpoint_name + '.index',
        last_checkpoint_name + '.meta',
        last_checkpoint_name + '.data-00000-of-00001'
    ]
    
    local_folder = os.path.join(output_dir, run_id)
    os.makedirs(local_folder, exist_ok=True)
    for filename in checkpoint_files:
        local_path = os.path.join(local_folder, filename)
        print(os.path.join(remote_checkpoint_dir, filename))
        sftp_client.get(os.path.join(remote_checkpoint_dir, filename), local_path)

def download_checkpoints(run_ids):
    with connect() as ssh_client, ssh_client.open_sftp() as sftp_client:
        for run_id in run_ids:
            print(f'Downloading: {run_id}')
            download_checkpoint(ssh_client, sftp_client, run_id)

if __name__ == '__main__':
    download_checkpoints(sys.argv[1:])