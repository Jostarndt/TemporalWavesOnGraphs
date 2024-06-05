import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import time  # Import time for timing

from config_loader import ConfigLoader
import models
import utils

def train(model, train_loader, validation_loader, optimizer, loss_func, device, config):
    start_time = time.time()  # Start timing
    
    logged_loss_val = []
    logged_loss_train = []
    targets_val_list = []
    pred_val_list = []
     
    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 5  # Number of epochs to wait before stopping
    
    # Training loop
    for epoch in range(config.num_epochs):
        targets_val_list = []
        pred_val_list = []
        model.train()
        running_loss = 0.0
        log_loss = 0
        for batch_idx, data in enumerate(train_loader):
            inputs, targets = data
            inputs, targets = torch.permute(inputs, (2, 1, 0)).to(device), torch.permute(targets, (2, 1, 0)).to(device)
            optimizer.zero_grad()
            
            outputs, _ = model(inputs, None, None, None, None, None, None, targets, train_mask="Yes")

            outputs = outputs[:, :targets.shape[1], :]
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 100 == 99:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Train Loss: {running_loss / 100:.4f}")
                log_loss += running_loss
                running_loss = 0.0
        logged_loss_train.append(log_loss)

        # Validation loop
        model.eval()
        running_loss_val = 0.0
        log_loss_val = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader):
                inputs, targets = data
                inputs, targets = torch.permute(inputs, (2, 1, 0)).to(device), torch.permute(targets, (2, 1, 0)).to(device)
                
                outputs, _ = model(inputs, None, None, None, None, None, None, None, train_mask="Yes")
                outputs = outputs[:, :targets.shape[1], :]
                val_loss = loss_func(outputs, targets)
                running_loss_val += val_loss.item()
                targets_val_list.append(targets.cpu())
                pred_val_list.append(outputs.cpu())
        
                if batch_idx % 100 == 99:  # Print every 100 batches
                    print(
                        f"Epoch {epoch+1}, Batch {batch_idx+1}, Val-Loss: {running_loss_val / 100:.4f}")
                    log_loss_val += running_loss_val
                    running_loss_val = 0.0
            logged_loss_val.append(log_loss_val)      

            if log_loss_val < best_val_loss:
                best_val_loss = log_loss_val
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    break  
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    # Plotting
    targets_val = torch.stack(targets_val_list)
    pred_val = torch.stack(pred_val_list)

    utils.branch_plot(targets_val, pred_val, llv=logged_loss_val[-1], name_string=config.output_name, output_dir=config.output_name)
    utils.training_loss_plot(logged_loss_train, logged_loss_val, name_string=config.output_name, output_dir=config.output_name)

    torch.save(model, os.path.join(config.output_name, config.output_name))


def main():
    # Load configuration and setup
    config = ConfigLoader("rnn.yml").get_config()
    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")

    # Data loading and training
    data_sources_list = config.datasources
    for data_sources in data_sources_list:
        datasets = [utils.get_dataset(data_source, config, train=True) for data_source in data_sources]
        train_dataset = ConcatDataset([torch.utils.data.Subset(ds, range(0, int(0.8 * len(ds)))) for ds in datasets])
        validation_dataset = ConcatDataset([torch.utils.data.Subset(ds, range(int(0.8 * len(ds)), len(ds))) for ds in datasets])

        # Training runs
        for run in range(config.num_train_runs):
            # Modify output name to include run number
            config.output_name = f"{config.model_name}_{run}_" + '_'.join([ds[:4] for ds in data_sources])
            os.makedirs(config.output_name, exist_ok=True)

            # Model setup for each run to ensure it starts fresh
            model = models.EncoderDecoder(
                input_size=1,
                enc_hidden_size=config.rnn_hidden_size,
                encoded_feat_size=0,
                dec_hidden_size=config.rnn_hidden_size,
                out_size=config.forecast_length,
                static_feat_size=0,
                enc_dropout=config.dropout,
                dec_dropout=config.dropout,
                enc_num_layers=config.rnn_num_layers,
                dec_num_layers=config.rnn_num_layers,
                embedding_size=0,
                device=device).to(device)

            if 'simulation' in data_sources and len(data_sources) > 1: # Load pre-trained model
                datasets = [utils.get_dataset(data_source, config, train=True) for data_source in [data_sources[1]]]
                train_dataset = ConcatDataset([torch.utils.data.Subset(ds, range(0, int(0.8 * len(ds)))) for ds in datasets])
                validation_dataset = ConcatDataset([torch.utils.data.Subset(ds, range(int(0.8 * len(ds)), len(ds))) for ds in datasets])
                model_dir = f"{config.model_name}_{run}_" + 'simu'
                model_path = os.path.join(model_dir, model_dir)
                if not os.path.exists(model_path):
                    print(f"Model path {model_path} does not exist. Skipping.")
                    continue

                model = torch.load(model_path).to(device)
                model.device = device
            
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of trainable model parameters: {total_params}")      

            loss_func = models.RMSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            
            # DataLoader setup for each run
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

            # Train the model for this run
            train(model, train_loader, validation_loader, optimizer, loss_func, device, config)

if __name__ == "__main__":
    main()    