from pathlib import Path


class DatasetConfiguration:

    def __init__(self,
                 dataset_name,
                 train_images_path,
                 train_labels_path,
                 test_images_path,
                 test_labels_path,
                 test_masks_path,
                 validation_images_path,
                 validation_labels_path,
                 image_width=None,
                 image_height=None):
        self.dataset_name = dataset_name
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path
        self.test_images_path = test_images_path
        self.test_labels_path = test_labels_path
        self.test_masks_path = test_masks_path
        self.val_images_path = validation_images_path
        self.val_labels_path = validation_labels_path
        self.image_width = image_width
        self.image_height = image_height

    @staticmethod
    def get_datasets_configuration(root):
        dataset_configs = {}
        root = Path(root)
        for dataset in ["DRIVE", "STARE", "CHASE", "DROPS", "HRF","UNI"]:
            troot = root / "eval" / dataset
            dataset_config = DatasetConfiguration(
                dataset_name=dataset + "-eval", 
                train_images_path=str(troot / "images"),
                train_labels_path=str(troot / "labels"),
                test_images_path=str(troot / "images"),
                test_labels_path=str(troot / "labels"),
                test_masks_path=str(troot / "masks"),
                validation_images_path=str(troot / "v" / "images"),
                validation_labels_path=str(troot / "v" / "labels"),
            )
            dataset_configs[dataset  + "-eval"] = dataset_config
            troot = root / "train/saunet" / dataset
            dataset_config = DatasetConfiguration(
                dataset_name=dataset,
                train_images_path=str(troot / "train/images"),
                train_labels_path=str(troot / "train/labels"),
                test_images_path=str(troot / "test/images"),
                test_labels_path=str(troot / "test/labels"),
                test_masks_path=str(troot / "test/masks"),
                validation_images_path=str(troot / "validate/images"),
                validation_labels_path=str(troot / "validate/labels"),
            )
            dataset_configs[dataset] = dataset_config
        return dataset_configs


DATASETS_ROOT_DIR = "./data"
datasets = DatasetConfiguration.get_datasets_configuration(DATASETS_ROOT_DIR)