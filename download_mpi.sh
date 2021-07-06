echo Fetching code from github...
svn export --force https://github.com/google-research/google-research/trunk/single_view_mpi

echo
echo Fetching trained model weights...
rm single_view_mpi_full_keras.tar.gz
rm -rf single_view_mpi_full
wget https://storage.googleapis.com/stereo-magnification-public-files/models/single_view_mpi_full_keras.tar.gz
tar -xzvf single_view_mpi_full_keras.tar.gz
rm single_view_mpi_full_keras.tar.gz
