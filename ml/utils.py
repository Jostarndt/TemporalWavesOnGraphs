import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt
import geopandas
import torch_geometric
import networkx as nx
import dataset
import os

#Plotting utilities
def branch_plot(targets, predictions, llv=0, stride = 1, name_string="", output_dir=""):
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    fig.suptitle(f"Val loss: {llv}")
    # stuttgart, berlin, barnim, goslar
    indices = [0, 140, 145, 200] if targets.shape[1] == 400 else [0, 5, 10, 20] 
    for en_index, index in enumerate(indices):
        assert (targets.shape[2] > stride)
        branch_target = torch.cat((targets[:,index, :stride, :].flatten(), targets[-1, index, 1:, :].squeeze()), dim = 0)

        x_axis = torch.tensor(np.arange(branch_target.shape[0]))

        axs[en_index].plot(x_axis, branch_target.cpu())
        #axs[en_index].set_xlabel(str(index) + " over time")
        axs[en_index].set_ylabel(f"Infection on {index}")
        axs[en_index].tick_params(labelrotation=45)
        
        pred_len = predictions.shape[2]
        for i in range(predictions.shape[0]):
            #print(f"shape of x_axis: {x_axis[i: i + pred_len].shape}")
            #print(f"shape of branch: {predictions[i, en_index, :, 0].shape}")
            axs[en_index].plot(x_axis[i: i + pred_len], predictions[i, index, :, 0].cpu())
        if (en_index < 3):
            axs[en_index].tick_params(axis="x",which="both", bottom=False, top=False, labelbottom=False)
    filepath = os.path.join(output_dir, f"branch_plot_{name_string}.png")
    plt.savefig(filepath, dpi = 600)
    plt.close()

        

def box_plot():
    pass

def training_loss_plot(train, validation, name_string="", output_dir=""):
    x = range(len(train))
    fig, ax1 = plt.subplots()
    ax1.plot(x, train, color="#E2725B")
    ax1.set_xlabel("Epochs")
    ax1.set_yscale("log")
    ax1.set_ylabel("Train Loss")
    ax1.text(len(train)-1, train[-1], f'{train[-1]:.4f}', ha='left', va='center', color="#E2725B")

    ax2 = ax1.twinx()

    ax2.plot(x, validation, color="#7F00FF")
    ax2.set_yscale("log")
    ax2.set_ylabel("Validation Loss")
    ax2.text(len(validation)-1, validation[-1], f'{validation[-1]:.4f}', ha='left', va='center', color="#7F00FF")

    '''
    plt.plot(train, label = "Train Loss")
    plt.plot(validation, label = "Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    '''
    #plt.legend()
    filepath = os.path.join(output_dir, f"loss_plot_{name_string}.png")
    plt.savefig(filepath)
    plt.close()


def plot_graph(val, pred, name_string, output_dir=""):
    dist = torch.load("./nuts3_adjacent_distances").T
    edge_index = dist[:2, :].int()
    dist = (1/dist[2,:])

    # next tile
    path = "/srv/data/csvs/meta_data/shapefiles/5000_NUTS3.shp" #TODO fix this
    gpd = geopandas.read_file(path).to_crs(epsg=4326)
    gpd['lon'] = gpd['geometry'].to_crs(epsg=4326).centroid.x
    gpd['lat'] = gpd['geometry'].to_crs(epsg=4326).centroid.y
    pos = gpd.set_index('NUTS_CODE')[['lon','lat']].T.to_dict('list')
    
    name_dict = gpd[gpd["NUTS_CODE"]!="DEG0N"]["NUTS_CODE"].reset_index()
    name_dict = name_dict.to_dict()

    t_ones = torch.ones([400])
    G = torch_geometric.data.Data(x=t_ones, edge_index=edge_index, edge_weight=dist)
    G.edge_attr = G.edge_weight

    G = torch_geometric.utils.to_networkx(G, edge_attrs=["edge_attr"])
    G = G.to_undirected()
    weights = list(nx.get_edge_attributes(G,"edge_attr").values())
    weights = [a * 1.e5 for a in weights ]
    
    G = nx.relabel_nodes(G, name_dict["NUTS_CODE"])
    
    fig, axs = plt.subplots(2, 3,figsize=(10, 8))
    stepsize = (val.shape[1]//2)-1
    for i in range(1, val.shape[1], stepsize):
        #for i in range(1,14,6):# (1, 28,13):# 
        nx.draw_networkx(G,node_size=50, vmin=0.0, vmax=0.5, with_labels=False,alpha=1,width=weights,node_color=val[:,i,:].squeeze().cpu(),edge_color=(0.27, 0.1, 0.01),pos = pos, ax = axs[0, i//stepsize])
        nx.draw_networkx(G,node_size=50, vmin=0.0, vmax=0.5, with_labels=False,alpha=1,width=weights,node_color=pred[:,i,:].squeeze().cpu(),edge_color=(0.27, 0.1, 0.01),pos = pos, ax = axs[1, i//stepsize])
        axs[1, i//stepsize].set_xlabel(f"t = {i}")

    axs[0, 0].set_ylabel("Ground Truth")
    axs[1, 0].set_ylabel("Prediction")

    #fig.colorbar(cmp, cax=cax)#, shrink=0.6)#, ax=axs[:,-1], shrink=0.6)

    #plt.figlegend()

    filepath = os.path.join(output_dir, f"{name_string}_Graph")
    plt.savefig(filepath)
    plt.close()

def daily_loss_plot(daily_loss, name_string, output_dir):
    filepath = os.path.join(output_dir, f"{name_string}_daily_loss")
    plt.plot(daily_loss.cpu())
    plt.savefig(filepath)
    plt.close()

class add_noise_transform:
    def __init__(self, mean = 0, std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        out = tensor + torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(out, min=0)
        #return tensor + torch.randn(tensor.size()) * self.std + self.mean
    def __repr__(self):
        return f"add_noise_transform, mean={self.mean}, std={self.std}"     

class remove_locations_transform:
    def __init__(self, perc = 1.):
        self.abs_drop = int(perc*400)
    def __call__(self, tensor):
        indices = torch.randperm(400)[:self.abs_drop]
        tensor[:, indices] = 0
        return tensor
    def __repr__(self):
        return f"randomly remove some entries" 

def get_dataset(key, config, train):
    noise_config = config.noise.get('train', {}) if train else config.noise.get('eval', {})
    
    mean = noise_config.get('gaussian_mean', 0)
    std = noise_config.get('gaussian_std', 0.01)
    locations_perc = noise_config.get('locations_perc', 1.)

    remove_transform = remove_locations_transform(locations_perc)
    noise_transform = add_noise_transform(mean, std)
    combined_transform = lambda x: remove_transform(noise_transform(x))
    
    dataset_map = {
        'simulation': dataset.SimulationDataset,
        'denoising': dataset.SimulationDenoisingDataset,
        'wave': dataset.WaveDataset,
        'advection': dataset.AdvectionDiffusionDataset
    }
    return dataset_map[key](config.encoder_length, config.forecast_length, transform=combined_transform)

def get_adjacency(key, device):
    if key == 'brazil_covid_adm1':
        filename = 'adm1_adjacent_distances' 
    elif key == 'wave':
        filename = 'germany_coastline_adjacency'
    else:
        filename = 'nuts3_adjacent_distances'


    adjacent_distances = torch.load(os.path.join('..', 'data', 'adjacency', filename)).T.to(device)
    edge_index = adjacent_distances[:2, :].int()
    dist = (1/adjacent_distances[2,:])
    return dist, edge_index
        