import glob
import mlflow
import os

from pathlib import Path

from utils.config import DatasetConfiguration

project_path = Path('.').resolve()
models_root_dir = project_path / 'models'
results_root_dir = project_path / 'results'
datasets_root_dir = project_path / 'data'

def predict_all_models():
    datasets_configs = DatasetConfiguration.get_datasets_configuration(datasets_root_dir)

    for i in ["DRIVE", "CHASE", "STARE"]:
        dcfg = datasets_configs[i + "-eval"]
        model_path = f"{models_root_dir}/{i}-model/weights.hdf5"
        parameters = {
            'model_path': model_path,
            'test_images_dir': str(project_path / dcfg.test_images_path),
            'test_labels_dir': str(project_path / dcfg.test_labels_path),
            'test_masks_dir': str(project_path / dcfg.test_masks_path),
            'dataset': i + "-eval",
            'output_dir': str(project_path / results_root_dir / i),
        }

        try:
            mlflow.projects.run(
                uri=str(project_path),
                entry_point='test',
                parameters=parameters,
                experiment_name='IterNet',
                use_conda=False
            )
        except mlflow.exceptions.ExecutionException as e:
            print('mlflow run execution failed.')
            print(e)
            pass

def predict_uni_models():
    datasets_configs = DatasetConfiguration.get_datasets_configuration(datasets_root_dir)

    for i in ["DRIVE", "CHASE", "STARE"]:
        dcfg = datasets_configs[i + "-eval"]
        model_path = f"{models_root_dir}/UNI-model/weights.hdf5"
        parameters = {
            'model_path': model_path,
            'test_images_dir': str(project_path / dcfg.test_images_path),
            'test_labels_dir': str(project_path / dcfg.test_labels_path),
            'test_masks_dir': str(project_path / dcfg.test_masks_path),
            'dataset': i + "-eval",
            'output_dir': str(project_path / results_root_dir / i),
        }

        try:
            mlflow.projects.run(
                uri=str(project_path),
                entry_point='test',
                parameters=parameters,
                experiment_name='IterNet',
                use_conda=False
            )
        except mlflow.exceptions.ExecutionException as e:
            print('mlflow run execution failed.')
            print(e)
            pass

if __name__ == '__main__':
    predict_uni_models()
