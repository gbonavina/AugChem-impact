import optuna
from gridsearch import GridSearch

def objective(trial):
    grid = GridSearch()

    augment_percentage = trial.suggest_float('augment_percentage', 0.1, 1.0)
    mask_ratio = trial.suggest_float('mask_ratio', 0.0, 0.5)
    delete_ratio = trial.suggest_float('delete_ratio', 0.0, 0.5)

    method_choices = [",".join(m) for m in grid.param_grid['augmentation_methods']]
    methods_str = trial.suggest_categorical('augmentation_methods', method_choices)
    methods = methods_str.split(",")

    params = {
        'augmentation_methods': methods,
        'augment_percentage': augment_percentage,
        'mask_ratio': mask_ratio,
        'delete_ratio': delete_ratio
    }

    print("Testando parâmetros:", params)
    augmented_data = grid.apply_augmentation(params)
    result = grid.train_model_on_dataset(augmented_data)
    print("Resultado:", result)
    if result['status'] == 'success':
        return result['metrics']['losses']['test_loss']
    else:
        print("Pruned:", params)
        raise optuna.TrialPruned()

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)  # Altere n_trials conforme necessário