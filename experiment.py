from gridsearch import GridSearch

# Ta dando um erro na execução pq os dados gerados pelo augchem sao para CL, (label nula), vou criar
# uma flag na lib pro usuario selecionar se ele vai usar pra CL ou nao, acho que isso resolve o problema.

test_config = {
            'epochs': 5,
            'batch_size': 128, 
            'lr': 0.00038419788129315696,
            'dropout': 0.3520475751955372,
            'weight_decay': 1.0528482325186746e-06,
            'num_layers': 2,
            'hidden_dim': 256,
            'embedding_dim': 64
        }

grid = GridSearch(
    dataset_path='QM9.csv',
    output_dir='test_grid_search',
    model_config=test_config
)

results = grid.run_grid_search()