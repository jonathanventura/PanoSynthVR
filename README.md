# PanoSynthVR
PanoSynthVR: View Synthesis From A Single Input Panorama with Multi-Cylinder Images

# Conda environment

To create and activate the conda environment:

    conda env create
    conda activate panosynthvr

Then run download_mpi.sh to get the MPI code and pre-trained weights.


Run MCI generation only
```
  python generate_mci.py --input example.jpg --width 2048 --height 1024 --o outputfolder
```
Run MCI generation only and display output in website using WebXR
```
  python generate_mci.py --input example.jpg --width 2048 --height 1024 --o outputfolder --s 1
```
