import torch


def config_map(options, model):
    optimizer = None
    scheduler = None
    
    
    if options.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    
    
    if options.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,                                                   threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        
        
    return optimizer, scheduler