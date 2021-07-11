#
# download_lila_subset.py
#
# Example of how to download a list of files from LILA, e.g. all the files
# in a data set corresponding to a particular species.
#

# %% Constants and imports

# import json
import os
import tempfile
import urllib.request
import zipfile
from multiprocessing.pool import ThreadPool
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm


# %% Support functions

def download_url(f_url, destination_filename=None, force_download=False, verbose=True):
    """
    Download a URL (defaulting to a temporary file)
    """

    if destination_filename is None:
        temp_dir = os.path.join(tempfile.gettempdir(), 'lila')
        os.makedirs(temp_dir, exist_ok=True)
        url_as_filename = f_url.replace('://', '_').replace('.', '_').replace('/', '_')
        destination_filename = \
            os.path.join(temp_dir, url_as_filename)

    if (not force_download) and (os.path.isfile(destination_filename)):
        print('Bypassing download of already-downloaded file {}'.format(os.path.basename(f_url)))
        return destination_filename

    if verbose:
        print('Downloading file {} to {}'.format(os.path.basename(f_url), destination_filename), end='')

    os.makedirs(os.path.dirname(destination_filename), exist_ok=True)
    urllib.request.urlretrieve(f_url, destination_filename)
    assert (os.path.isfile(destination_filename))

    if verbose:
        num_bytes = os.path.getsize(destination_filename)
        print('...done, {} bytes.'.format(num_bytes))

    return destination_filename


def download_relative_filename(f_url, output_base, verbose=False):
    """
    Download a URL to output_base, preserving relative path
    """

    parsed_url = urlparse(f_url)
    # remove the leading '/'
    assert parsed_url.path.startswith('/')
    relative_filename = parsed_url.path[1:]
    destination_filename = os.path.join(output_base, relative_filename)
    download_url(f_url, destination_filename, verbose=verbose)


def unzip_file(input_file, output_folder=None):
    """
    Unzip a zipfile to the specified output folder, defaulting to the same location as
    the input file
    """

    if output_folder is None:
        output_folder = os.path.dirname(input_file)

    with zipfile.ZipFile(input_file, 'r') as zf:
        zf.extractall(output_folder)


# LILA camera trap master metadata file
metadata_url = 'http://lila.science/wp-content/uploads/2020/03/lila_sas_urls.txt'

# We'll write images, metadata downloads, and temporary files here
output_dir = r'E:\ss_data'
os.makedirs(output_dir, exist_ok=True)

# We will demonstrate two approaches to downloading, one that loops over files
# and downloads directly in Python, another that uses AzCopy.
# AzCopy will generally be more performant and supports resuming if the
# transfers are interrupted.  It assumes that azcopy is on the system path.
use_azcopy_for_download = True

overwrite_files = False

# Number of concurrent download threads (when not using AzCopy) (AzCopy does its
# own magical parallelism)
n_download_threads = 50

# %% Download and parse the metadata file
# Put the master metadata file in the same folder where we're putting images
p = urlparse(metadata_url)
metadata_filename = os.path.join(output_dir, os.path.basename(p.path))
download_url(metadata_url, metadata_filename)

# Read lines from the master metadata file
with open(metadata_filename, 'r') as f:
    metadata_lines = f.readlines()
metadata_lines = [s.strip() for s in metadata_lines]

# Parse those lines into a table
metadata_table = {}
for s in metadata_lines:

    if len(s) == 0 or s[0] == '#':
        continue

    # Each line in this file is name/sas_url/json_url
    tokens = s.split(',')
    assert len(tokens) == 3
    url_mapping = {'sas_url': tokens[1], 'json_url': tokens[2]}
    metadata_table[tokens[0]] = url_mapping

    assert 'https' not in tokens[0]
    assert 'https' in url_mapping['sas_url']
    assert 'https' in url_mapping['json_url']

# In this example, we're using the Missouri Camera Traps data set and the Caltech Camera Traps dataset
datasets_of_interest = ['Snapshot Serengeti']
# All lower-case; we'll convert category names to lower-case when comparing
species_of_interest = ['aardwolf']
locations_of_interest = ['F04']
seasons_of_interest = ['SER_S9']

# %% List of files we're going to download (for all data sets)

# Flat list or URLS, for use with direct Python downloads
urls_to_download = []

# For use with azcopy
downloads_by_dataset = {}

for ds_name in datasets_of_interest:

    json_filename = str(r'C:\Users\mfarj\Documents\ss_data\SnapshotSerengeti_S1-11_v2.1.json')
    csv_filename = str(r'C:\Users\mfarj\Documents\ss_data\SnapshotSerengeti_v2_1_images.csv')
    sas_url = metadata_table[ds_name]['sas_url']

    base_url = sas_url.split('?')[0]
    assert not base_url.endswith('/')

    sas_token = sas_url.split('?')[1]
    assert not sas_token.startswith('?')

    print('Reading csv file...')
    df_images = pd.read_csv(csv_filename)
    print('...done')
    df_images.rename(columns={'Unnamed: 0': 'seq_id'}, inplace=True)
    print('HEAD: ', df_images.head())
    df_images['season'] = df_images.capture_id.map(lambda x: x.split('#')[0])
    print('Season counts: ', df_images.season.value_counts().sort_index())

    # To use for Json file download
    # Open the metadata file
    # print('Reading json file...')
    # with open(json_filename, 'r') as f:
    #     data = json.load(f)
    # print('...done')
    # df_images = pd.json_normalize(data['images'])
    # print('HEAD: ', df_images.head())
    # print('Done reading json file')
    # df_images['season'] = df_images.seq_id.map(lambda x: x.split('#')[0])
    # print('Season counts: ', df_images.season.value_counts().sort_index())

    filenames = None
    # Build a list of image files (relative path names) that match the target locations
    for season in seasons_of_interest:
        # Retrieve image file names that match the location
        filenames = df_images.loc[lambda x: x.season == season].image_path_rel
        print(f'Found {len(filenames)} images captured in {season} season.')
        # Convert to URLs
        for fn in filenames:
            url = base_url + '/' + fn
            urls_to_download.append(url)

    downloads_by_dataset[ds_name] = {'sas_url': sas_url, 'filenames': filenames}
    # ...for each dataset

print('Found {} images to download'.format(len(urls_to_download)))

# %% Download those image files
if use_azcopy_for_download:
    for ds_name in downloads_by_dataset:

        print('Downloading images for {} with azcopy'.format(ds_name))
        sas_url = downloads_by_dataset[ds_name]['sas_url']
        filenames = downloads_by_dataset[ds_name]['filenames']
        # We want to use the whole relative path for this script (relative to the base of the container)
        # to build the output filename, to make sure that different data sets end up in different folders.
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

        print('...done. Writing list of files to be downloaded...')
        # Write out a list of files, and use the azcopy "list-of-files" option to download those files this azcopy
        # feature is unofficially documented at
        # https://github.com/Azure/azure-storage-azcopy/wiki/Listing-specific-files-to-transfer
        az_filename = os.path.join(output_dir, 'filenames_{}.txt'.format(ds_name.lower().replace(' ', '_')))
        with open(az_filename, 'w') as f:
            for fn in filenames:
                f.write(fn.replace('\\', '/') + '\n')

        print('...done. Downloading with azcopy...')
        cmd = 'azcopy cp "{0}" "{1}" --list-of-files "{2}"'.format(
            container_sas_url, container_output_dir, az_filename)
        # import clipboard; clipboard.copy(cmd)
        os.system(cmd)
else:
    # Loop over files
    print('Downloading images for {0} without azcopy'.format(species_of_interest))
    if n_download_threads <= 1:
        for url in tqdm(urls_to_download):
            download_relative_filename(url, output_dir, verbose=True)
    else:
        pool = ThreadPool(n_download_threads)
        tqdm(pool.imap(lambda s: download_relative_filename(s, output_dir, verbose=False), urls_to_download),
             total=len(urls_to_download))
print('Done!')
