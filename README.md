# Epedemic PDE simulator

## Reference
DOI: ? Bibtex: ? 

## Execution

### Requirements
Installation of *deal.ii* using official 
[deal.ii installation](https://dealii.org/download.html). As this is tested on 
Ubuntu, this can ?? be done with

```
apt-get install libdeal.ii-dev
```


### Running the Simulation

1. ``` cd simulation ```
2. ``` cmake . ```
3. ``` make run ```

### Using the results

Running the simulation as described above will result in multiple CSVs and .vtk
 files, named after the simulation and its respective timestep. VKT files can be
 visualized with 
[ParaView](https://gitlab.kitware.com/paraview/paraview/-/tree/master). The CSVs
 have to be concatenated to form a new dataset.

## Data Usage (without execution)
The data can be loaded and used with only numpy.
Executing the following numpy code

```
import numpy as np
a = np.load("364_25_numpy_dataset.npy")
print(a.shape)
```
will reveal the dataset has a shape of ``` (9100, 400, 2) ``` which refers
to 9100 timesteps, 400 locations/nodes/NUTS3 regions, and two values: 
Susceptible and Infected.

The Nodes are the alphabetically ordered NUTS3 regions. Their 
adjacencies and distances can be found in the file *nuts3_adjacent_distances", 
which can directly loaded as a PyTorch tensor with the
following code:


```
dist = torch.load("nuts3_adjacent_distances").T
edge_index = dist[:2, :].int()
dist = dist[2,:]
```

will return an edge_index of 2088 connection, and their distances. 



## License
