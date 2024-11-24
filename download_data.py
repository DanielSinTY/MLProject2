import urllib.request
from getfilelistpy import getfilelist
from os import path, makedirs, remove, rename

def download_googledrive_folder(remote_folder, local_dir, gdrive_api_key):
    """
    Download a folder from Google Drive

    Parameters
    ----------
    remote_folder : str
        URL of the Google Drive folder
    local_dir : str
        Local directory to save the files
    gdrive_api_key : str
        Google Drive API key
    
    Returns
    -------
    bool
        True if the download is successful, False otherwise

    Sample Usage
    ------------
    download_googledrive_folder('https://drive.google.com/drive/folders/1as-3mQJ91XvDOEHBlwYHwz41j_p14jR-?usp=sharing', 'data', os.environ['API_KEY'])
    """


    success = True

    
    try:
        resource = {
            "api_key": gdrive_api_key,
            "id": remote_folder.split('/')[-1].split('?')[0],
            "fields": "files(name,id)",
        }
        res = getfilelist.GetFileList(resource)
        print('Found #%d files' % res['totalNumberOfFiles'])
        destination = local_dir
        if not path.exists(destination):
            makedirs(destination)

        for i in range(1, res['totalNumberOfFolders']):
            destination_directory = destination
            for j in range(1, len(res['fileList'][i]['folderTree'])):
                folder_idx =  res['folderTree']['folders'].index(res['fileList'][i]['folderTree'][j])
                destination_directory = path.join(destination_directory, res['folderTree']['names'][folder_idx])
                if not path.exists(destination_directory):
                    makedirs(destination_directory)
            
            for file_dict in res['fileList'][i]['files']:
                if not file_dict['name'].endswith('.csv'):
                    continue
                print('Downloading %s' % file_dict['name'])
                # if gdrive_api_key:
                #     source = "https://www.googleapis.com/drive/v3/files/%s?alt=media&confirm=t&key=%s" % (file_dict['id'], gdrive_api_key)
                # else:
                source = "https://drive.google.com/uc?id=%s&export=download" % file_dict['id']  # only works for small files (<100MB)
                
                destination_file = path.join(destination_directory, file_dict['name'])
                urllib.request.urlretrieve(source, destination_file)

    except Exception as err:
        print(err)
        success = False

    return success
