from gridsearch import GridSearch

test_config = {
            'epochs': 250,
            'batch_size': 32, 
            'lr': 0.00038419788129315696,
            'dropout': 0.3520475751955372,
            'weight_decay': 1.0528482325186746e-06,
            'num_layers': 2,
            'hidden_dim': 256,
            'embedding_dim': 16
        }

grid = GridSearch(
    dataset_path='QM9.csv',
    output_dir='test_grid_search',
    model_config=test_config
)

results = grid.run_grid_search()