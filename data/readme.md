The directory contains data for testing, demonstrations, and experiments. 

`sync` - It's expected that this directory is in sync with the [Google Drive sync](https://drive.google.com/drive/u/0/folders/113-F2kbpGSFJ3JyDqKTLNT_ablVrjs7b) 
directory at the moment of script execution. The directory is ignored by git. 

Setup Google Drive synchronization service for downloading `sync` folder with your machine and use os 
symlink for reference it from the repository.   

Windows: `mklink /D data\sync "C:\your_google_drive_path\sync"` from `cmd` with admin rights


`local` - Use this directory to store temporary local files that script can produce or download and which
shouldn't be synced. The directory is ignored by git.
