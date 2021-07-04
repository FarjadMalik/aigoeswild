from urllib.request import urlopen
#from io import BytesIO
#from zipfile import ZipFile
from tempfile import NamedTemporaryFile
from shutil import unpack_archive

zip_url_path = 'https://lilablobssc.blob.core.windows.net/snapshotserengeti-v-2-0/SnapshotSerengeti_S09_v2_0_part1.zip'
store_path = r'E:\ss_data_v2'


def download_and_unzip(url, extract_to='.'):
    print(f'---download_and_unzip--')
    print(f'Opening url: {url}')
    # with urlopen(url) as zipresp:
    #     print(f'Reading url into zip file')
    #     with ZipFile(BytesIO(zipresp.read())) as zfile:
    #         print(f'Unpacking to {extract_to}')
    #         zfile.extractall(extract_to)
    #Storing in a temp file way
    with urlopen(url) as zipresp, NamedTemporaryFile() as tfile:
        tfile.write(zipresp.read())
        tfile.seek(0)
    print(f'Successfully loaded into a temp file. Unpacking...')
    unpack_archive(tfile.name, extract_to, format='zip')
    print(f'---return--')


if __name__ == '__main__':
    print(f'---download_extract_zipfile.py---')
    download_and_unzip(zip_url_path, store_path)
