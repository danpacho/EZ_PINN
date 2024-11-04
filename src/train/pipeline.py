from dataclasses import asdict, dataclass
import json
import os
import statistics
import torch

from torch.utils.data import Dataset
from typing import Literal, Union

from tqdm.auto import tqdm
from src.train.trainer import Trainer
from src.train.model_component import ModelComponent

from torch.utils.data import DataLoader

from src.train.uuid_encoder import UUIDEncoder
from src.train.storage.storage_unit import KeyValueStorage


class Pipeline:
    @property
    def base_path(self) -> str:
        """
        Get the base path.

        Returns:
            str: default(`$pipeline$`)
        """
        return self._base_path

    def change_base_path(self, new_base_path: str) -> None:
        """
        Change the base path.

        Args:
            path (str): New path
        """
        self._log("Storage", f"Changing base path to {new_base_path}")
        is_prev_path_data_exists = os.path.exists(
            f"{self.base_path}/{self.pipeline_name}"
        )

        if is_prev_path_data_exists:
            self._log("Storage", f"Moving data to {new_base_path}/{self.pipeline_name}")
            os.rename(
                f"{self.base_path}/{self.pipeline_name}",
                f"{new_base_path}/{self.pipeline_name}",
            )

        self._base_path = new_base_path
        self._log("Storage", f"Base path changed to {new_base_path}")

    def __init__(
        self,
        trainer: Trainer,
        data_sets: tuple[Dataset, Dataset],
        pipeline_name: str,
        pipeline_store_base_path: str = "$pipeline$",
    ) -> None:
        """
        Initialize TrainPipeline.

        Args:
            model (Model): Model class, should be implemented by user.
            trainer (Trainer): Trainer class, should be implemented by user.
            data_loaders (tuple[DataLoader, DataLoader]): {train_loader, val_loader}
            pipeline_name (str): Name of the pipeline
            pipeline_store_base_path (str, optional): Base path to store the pipeline data. Defaults to "$PIPELINE$".
        """
        # Members
        self.trainer = trainer
        self.train_set, self.validation_set = data_sets
        self.train_loader = DataLoader(
            self.train_set, batch_size=trainer.train_config.batch_size, shuffle=True
        )
        self.validation_loader = DataLoader(
            self.validation_set,
            batch_size=trainer.train_config.batch_size,
            shuffle=False,
        )
        self.pipeline_name = pipeline_name

        # Default base path
        self._base_path = pipeline_store_base_path

        # UUID generator
        self.uuid_gen = UUIDEncoder(
            storage_file=f"{self._base_path}/{pipeline_name}/uuid.json",
        )
        self.project_root = f"{self.base_path}/{self.pipeline_name}/{self.uuid}"
        self.db = KeyValueStorage(
            label="PipelineDB",
            root_filename=f"{self._base_path}/{pipeline_name}/{self.uuid}/db.json",
        )

        # Setup
        self._setup()

    @property
    def model_component(self) -> ModelComponent:
        return self.trainer.model_component

    def load_trained_model(self) -> None:
        """
        Load the trained model.
        """
        self._log("Loading", "Loading trained model")
        self.model_component.load_model(f"{self.project_root}/model.pth")
        self._log("Loading", "Model loaded")

    @property
    def uuid(self) -> str:
        pipeline_seed = f"{self.model_component.model_seed}-{self.trainer.trainer_seed}"
        return self.uuid_gen.encode_to_uuid(pipeline_seed)

    def _setup(self) -> None:
        """
        Setup the training pipeline.

        1. Create `{project} = {base_path}/{db_name}/{uuid}` root
        2. Create `{project}/models` directory
        3. Print data information
        """
        self._log("Welcome", "Training pipeline initialized")

        project_root = self.project_root

        if not os.path.exists(project_root):
            self._log("Storage", f"Created project root: {project_root}")

            os.makedirs(project_root)

        self._log("Health Check", "Pipeline is ok to go")

        self._device = self.get_device()
        self._init_gpu()

        self._log("GPU Setup", f"device: {self._device}")

        self.model_component.model.to(self._device)

        if self._device.type == "cuda":
            gpu_num = torch.cuda.current_device()
            print("Current cuda device ", gpu_num)
            print(
                "Allocated:",
                round(torch.cuda.memory_allocated(gpu_num) / 1024**3, 1),
                "GB",
            )
            print(
                "Cached:   ",
                round(torch.cuda.memory_reserved(gpu_num) / 1024**3, 1),
                "GB",
            )
            print(torch.cuda.get_device_name(gpu_num))
            print("Available devices ", torch.cuda.device_count())
            print(f"CUDA version: {torch.version.cuda}")

        self._log("Trainer", "Dependencies injected")
        self.trainer.inject_dependencies(device=self._device)

    def get_device(self, gpu_num: int = 3) -> Union[str, Literal["cpu"]]:
        """
        Get device
        """
        device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
        return device

    def _init_gpu(self) -> None:
        torch.cuda.set_device(self._device)

    @property
    def train_size(self) -> int:
        """
        Get the train-set size.
        """
        return len(self.train_loader.dataset)

    @property
    def validation_size(self) -> int:
        """
        Get the validation-set size.
        """
        return len(self.validation_loader.dataset)

    def _log(self, title: str, text: str) -> None:
        top_header = f"{'-' * 25}[ {title.upper()} ]{'-' * 25}"
        print(f"{top_header}\n{text}\n{'-' * (len(top_header) + 2)}")

    def plot_loss_history(self) -> None:
        """
        Plot the training history.
        """
        import matplotlib.pyplot as plt

        self._log("Plotting", "Plotting loss history")

        loss_epoch_train = self.db.inquire("loss_epoch_train")
        loss_epoch_test = self.db.inquire("loss_epoch_test")

        if loss_epoch_train is None or loss_epoch_test is None:
            self._log("Plotting", "Data not found")
            return

        title = "loss"
        label_axis_x = "epoch"
        label_axis_y = "loss"

        plt.figure(figsize=(8, 6))
        plt.title(title)

        plt.plot(loss_epoch_train, "-", color="blue", label="train")
        plt.plot(loss_epoch_test, "-", color="red", label="test")
        plt.legend()

        plt.xlabel(label_axis_x)
        plt.ylabel(label_axis_y)

        plt.tight_layout()
        plt.show()

    def plot_accuracy_history(self) -> None:
        """
        Plot the accuracy history.
        """
        import matplotlib.pyplot as plt

        self._log("Plotting", "Plotting accuracy history")

        accuracy_epoch_train = self.db.inquire("accuracy_epoch_train")
        accuracy_epoch_test = self.db.inquire("accuracy_epoch_test")

        max_accuracy_test = max(accuracy_epoch_test)
        max_accuracy_train = max(accuracy_epoch_train)

        self._log("Plotting", f"Max accuracy (train): {max_accuracy_train}")
        self._log("Plotting", f"Max accuracy (test): {max_accuracy_test}")

        if accuracy_epoch_train is None or accuracy_epoch_test is None:
            self._log("Plotting", "Data not found")
            return

        title = "accuracy"
        label_axis_x = "epoch"
        label_axis_y = "accuracy"

        plt.figure(figsize=(8, 6))
        plt.title(title)

        plt.plot(accuracy_epoch_train, "-", color="blue", label="train")
        plt.plot(accuracy_epoch_test, "-", color="red", label="test")
        plt.legend()

        plt.xlabel(label_axis_x)
        plt.ylabel(label_axis_y)

        plt.tight_layout()
        plt.show()

    def plot_lr_history(self) -> None:
        """
        Plot the learning rate history.
        """
        import matplotlib.pyplot as plt

        self._log("Plotting", "Plotting learning rate history")

        lr_epoch_train = self.db.inquire("lr_epoch_train")

        if lr_epoch_train is None:
            self._log("Plotting", "Data not found")
            return

        title = "learning rate schedule"
        label_axis_x = "epoch"
        label_axis_y = "learning rate"

        plt.figure(figsize=(8, 6))
        plt.title(title)

        plt.plot(lr_epoch_train, "-", color="green")
        plt.xlabel(label_axis_x)
        plt.ylabel(label_axis_y)

        plt.tight_layout()
        plt.show()

    def train(self) -> None:
        """
        Train the model.
        """
        self._log("Training", "Training started")

        total_steps = len(self.train_loader) * self.trainer.train_config.epoch
        progress_bar = tqdm(total=total_steps, desc="Training")

        loss_epoch_train = []
        loss_epoch_test = []
        accuracy_epoch_train = []
        accuracy_epoch_test = []
        lr_epoch_train = []

        early_stop_tolerance = 0
        prev_best_loss = float("inf")

        for epoch in range(self.trainer.train_config.epoch):
            loss_iter_train = []
            loss_iter_test = []
            accuracy_iter_train = []
            accuracy_iter_test = []
            lr_iter_train = []

            # training loss and training accuracy
            for batch, train_data in enumerate(self.train_loader):
                (loss, accuracy, lr) = self.trainer.train(train_data)

                progress_bar.update(1)

                lr_iter_train.append(lr)
                loss_iter_train.append(loss)
                accuracy_iter_train.append(accuracy)

            lr_epoch_train = lr_epoch_train + lr_iter_train

            # testing loss and testing accuracy
            for batch, validation_data in enumerate(self.validation_loader):
                (loss, accuracy) = self.trainer.evaluate(validation_data)

                loss_iter_test.append(loss)
                accuracy_iter_test.append(accuracy)

            loss_mean_train = statistics.mean(loss_iter_train)
            loss_mean_test = statistics.mean(loss_iter_test)
            accuracy_train = statistics.mean(accuracy_iter_train)
            accuracy_test = statistics.mean(accuracy_iter_test)

            loss_epoch_train.append(loss_mean_train)
            loss_epoch_test.append(loss_mean_test)
            accuracy_epoch_train.append(accuracy_train)
            accuracy_epoch_test.append(accuracy_test)

            logs_print = {
                "epoch": epoch + 1,
                "loss(train)": loss_mean_train,
                "loss(test)": loss_mean_test,
                "accuracy(train)": accuracy_train,
                "accuracy(test)": accuracy_test,
                "tolerance": early_stop_tolerance,
                "lr": lr_epoch_train[-1],
            }
            progress_bar.set_postfix(**logs_print)

            torch.cuda.empty_cache()  # Empty the cache for GPU out of memory error

            if loss_mean_test < prev_best_loss:
                early_stop_tolerance = 0
                prev_best_loss = loss_mean_test
                self.model_component.save_model(f"{self.project_root}/model.pth")
            else:
                early_stop_tolerance += 1

            if early_stop_tolerance >= self.trainer.train_config.early_stopping:
                self._log("Training", "!Early stopping triggered")
                break

        self._log("Training", "Training completed")
        progress_bar.close()

        self._log("Storage", "Storing training history data...")
        self.db.insert_field("lr_epoch_train", lr_epoch_train)
        self.db.insert_field("loss_epoch_train", loss_epoch_train)
        self.db.insert_field("loss_epoch_test", loss_epoch_test)
        self.db.insert_field("accuracy_epoch_train", accuracy_epoch_train)
        self.db.insert_field("accuracy_epoch_test", accuracy_epoch_test)
        self.db.save()
        self._log("Storage", "Training history data stored")

    def evaluate(self) -> None:
        """
        Evaluate the model.
        """
        self._log("Evaluation", "Loading model...")
        try:
            self.model_component.load_model(f"{self.project_root}/model.pth")
            self._log("Evaluation", "Model loaded, evaluation started")
        except FileNotFoundError:
            self._log("Evaluation", "Model not found, evaluation stopped")
            return

        loss_iter_test = []
        accuracy_iter_test = []

        for batch, validation_data in enumerate(self.validation_loader):
            (loss, accuracy) = self.trainer.evaluate(validation_data)

            loss_iter_test.append(loss)
            accuracy_iter_test.append(accuracy)

        loss_mean_test = statistics.mean(loss_iter_test)
        accuracy_test = statistics.mean(accuracy_iter_test)

        self._log("Evaluation", "Evaluation completed")

        self._log("Storage", "Storing evaluation history data...")
        self.db.insert_field("loss_epoch_test_eval", loss_mean_test)
        self.db.insert_field("accuracy_epoch_test_eval", accuracy_test)
        self.db.save()
        self._log("Storage", "Evaluation history data stored")

    def experimental_log(self, title: str, text: str) -> None:
        """
        Log the experimental data.
        """
        self.db.insert_field(title, text)
        self.db.save()
