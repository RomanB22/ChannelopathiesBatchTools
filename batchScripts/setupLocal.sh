unset DISPLAY
conda activate M1_CEBRA
export PYTHONPATH=$PYTHONPATH:$PWD # do it in \src and in parent folder
cd src
export PYTHONPATH=$PYTHONPATH:$PWD # do it in \src and in parent folder
cd ..
nrnivmodl -coreneuron mod/