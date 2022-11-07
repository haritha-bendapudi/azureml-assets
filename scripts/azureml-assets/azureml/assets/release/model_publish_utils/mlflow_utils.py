# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""MLFlow Operations Class."""

import logging
from mlflow.models import ModelSignature
from transformers import AutoTokenizer, AutoConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)
import azureml.evaluate.mlflow as hf_mlflow


class MLFlowModelUtils:
    """Transform Model to MLFlow Model."""

    def __init__(self, name, task_name, flavor, mlflow_model_dir):
        """Initialize object for MLFlowModelUtils."""
        self.name = name
        self.task_name = task_name
        self.mlflow_model_dir = mlflow_model_dir
        self.flavor = flavor

    def _convert_to_mlflow_hftransformers(self):
        """Convert the model using MLFlow Huggingface Flavor."""
        # TODO Add MLFlow HFFlavour support once wheel files are publicly available
        config = AutoConfig.from_pretrained(self.name)
        misc_conf = {"task_type": self.task_name}
        task_model_mapping = {
            "multiclass": AutoModelForSequenceClassification,
            "multilabel": AutoModelForSequenceClassification,
            "fill-mask": AutoModelForMaskedLM,
            "ner": AutoModelForTokenClassification,
            "question-answering": AutoModelForQuestionAnswering,
            "summarization": AutoModelForSeq2SeqLM,
            "text-generation": AutoModelForCausalLM,
            "text-classification": AutoModelForSequenceClassification,
            "translation": AutoModelForSeq2SeqLM
        }
        if self.task_name in task_model_mapping:
            model = task_model_mapping[self.task_name].from_pretrained(self.name, config=config)
        elif "translation" in self.task_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.name, config=config)
        else:
            logging.error("Invalid Task Name")
        tokenizer = AutoTokenizer.from_pretrained(self.name, config=config)
        sign_dict = {
            "inputs": '[{"name": "input_string", "type": "string"}]',
            "outputs": '[{"type": "string"}]',
        }
        if self.task_name == "question-answering":
            sign_dict["inputs"] = '[{"name": "question", "type": "string"}, {"name": "context", "type": "string"}]'
        signature = ModelSignature.from_dict(sign_dict)

        try:
            hf_mlflow.hftransformers.save_model(
                model,
                f"{self.mlflow_model_dir}",
                tokenizer,
                config,
                misc_conf,
                signature=signature,
            )
        except Exception as e:
            logging.error("Unable to transform model as MLFlow model due to error: " + e)
        return None

    def _convert_to_mlflow_package(self):
        """Convert the model using pyfunc flavor."""
        return None

    def covert_into_mlflow_model(self):
        """Convert the model with given flavor."""
        if self.flavor == "hftransformers":
            self._convert_to_mlflow_hftransformers()
        # TODO add support for pyfunc. Pyfunc requires custom env file.
        else:
            self._convert_to_mlflow_package()
