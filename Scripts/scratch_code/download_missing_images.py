#
# download_lila_subset.py
#
# Example of how to download a list of files from LILA, e.g. all the files
# in a data set corresponding to a particular species.
#

# %% Constants and imports

# import json
import os
from urllib.parse import urlparse

# We'll write images, metadata downloads, and temporary files here
output_dir = r"C:\Users\mfarj\Documents\ss_data"
os.makedirs(output_dir, exist_ok=True)

sas_url = str('https://lilablobssc.blob.core.windows.net/snapshotserengeti-unzipped?st=2020-01-01T00%3A00%3A00Z&se'
              '=2034-01-01T00%3A00%3A00Z&sp=rl&sv=2019-07-07&sr=c&sig=/DGPd%2B9WGFt6HgkemDFpo2n0M1htEXvTq9WoHlaH7L4'
              '%3D')
base_url = sas_url.split('?')[0]
sas_token = sas_url.split('?')[1]
p = urlparse(base_url)
account_path = p.scheme + '://' + p.netloc
assert account_path == 'https://lilablobssc.blob.core.windows.net'

container_and_folder = p.path[1:]
container_name = container_and_folder.split('/')[0]
container_sas_url = account_path + '/' + container_name + '?' + sas_token
print('Creating containers for outputs...')

# The container name will be included because it's part of the file name
container_output_dir = os.path.join(output_dir, container_name)
os.makedirs(container_output_dir, exist_ok=True)

az_filename = os.path.join(output_dir, 'filenames_snapshot_serengeti_sample.txt')
print(f"container_sas_url: {container_sas_url}")
print(f"container_output_dir: {container_output_dir}")
print(f"az_filename: {az_filename}")

print('...done. Downloading with azcopy...')
cmd = 'azcopy cp "{0}" "{1}" --list-of-files "{2}"'.format(
    container_sas_url, container_output_dir, az_filename)
print(f"command: \n{cmd}")

os.system(cmd)
print('Done!')
