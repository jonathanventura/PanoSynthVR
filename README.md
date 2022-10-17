# PanoSynthVR
PanoSynthVR: View Synthesis From A Single Input Panorama with Multi-Cylinder Images

Code from our papers:

John Waidhofer, Richa Gadgil, Anthony Dickson, Stefanie Zollmann, and Jonathan Ventura.  PanoSynthVR: Toward Light-weight 360-Degree View Synthesis from a Single Panoramic Input.  IEEE International Symposium on Mixed and Augmented Reality. 2022.

Richa Gadgil, Reesa John, Stefanie Zollmann, and Jonathan Ventura.  PanoSynthVR: View Synthesis From A Single Input Panorama with Multi-Cylinder Images.  ACM SIGGRAPH 2021 Posters.

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
