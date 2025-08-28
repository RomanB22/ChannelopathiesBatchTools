# Primary motor cortex (M1) circuits model
## Description
Multiscale model of mouse primary motor cortex (M1) developed using NetPyNE (www.netpyne.org).

The model was used to benchmark CoreNEURON in the following paper:

- Awile O, Kumbhar P, Cornu N, Dura-Bernal S, Gonzalo JK, Lupton O, Magkanaris I, McDougal R, Newton AJH, Pereira A, Savulescu A, Carnevale NT, Hines M, Lytton WW, Schurmann F. **Modernizing the NEURON Simulator for Sustainability, Portability, and Performance**. Frontiers in Neuroinformatics (Under Revision). Research Topic: "Neuroscience, Computing, Performance, and Benchmarks: Why It Matters to Neuroscience How Fast We Can Compute."


Previous versions of the model were described in the following papers:

- Dura-Bernal S, Neymotin SA, Suter BA, Dacre J, Schiemann J, Duguid I, Shepherd GMG, Lytton WW. **Multiscale model of primary motor cortex circuits reproduces in vivo cell type-specific dynamics associated with behavior.** BioRxiv 2022.02.03.479040; doi: https://doi.org/10.1101/2022.02.03.479040. 

- Sivagnanam S, Gorman W, Doherty D, Neymotin SA, Fang S, Hovhannisyan H, Lytton WW, Dura-Bernal S. **Simulating large-scale models of brain neuronal circuits using Google Cloud Platform.** Practice and Experience in Advanced Research Computing, PEARC2020. 10.1145/3311790.3399621


This specific version of the model was extended to include two new types of interneurons -- NGF and VIP -- and has not been previously published.

## Download the Repository

To clone the dev branch of the repository, open a terminal in the directory where you'd like to store the project and run:

```bash
git clone --branch Manifolds --single-branch https://github.com/RomanB22/M1_NEURON_paper.git M1_Manifolds
````

After that, make sure to move to the repo folder using `cd M1_Manifolds`.

## Setup and execution

Requires NEURON with Python and MPI support. 

1. From parent folder run `nrnivmodl mod`. This should create a directory called x86_64.
2. Add project root directory to PYTHONPATH: `export PYTHONPATH=PYTHONPATH:$PWD`
3. To run type: `mpiexec -np [num_proc] nrniv -python -mpi src/init.py`

## Overview of file structure:

* /src/init.py: Main executable; calls functions from other modules. Sets what parameter file to use.

* /src/netParams.py: Network parameters

* /src/cfg.py: Simulation configuration

* /src/defs.py: Functions to define cell types, connections and auxiliar functions to setup the in-vivo spikes for the model. 

* /cells: source .py, .hoc, .json or .pkl files for the different cell types used in the model; these will be imported into netpyne

* /conns: source .py, .json or .pkl files for the network connectivity; these will be imported into netpyne

* /mod: NMODL files containing the ionic channel and synaptic mechanisms used in the model 

* /data: where the model data is stored 

* /manifolds: where the data related to manifolds is stored 

* /batchData: where the simulation data is stored 


For further information please contact: salvadordura@gmail.com 

