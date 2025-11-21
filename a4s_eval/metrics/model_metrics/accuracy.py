from datetime import datetime


from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.functional_model import TabularClassificationModel


@model_metric(name="accuracy")
def accuracy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: TabularClassificationModel,
) -> list[Measure]:
    # Both x and y (the features and the target) are contained in dataset.data as a dataframe.
    # To identify the target (y), use the datashape.target object, which has a name property. Use this property to index the aforementioned dataframe.
    # To identify the features (x), use the datashape.features list of object. Similarly each object in this list has a name property to index the dataframe.

    # Inspect FunctionalModel definition to identify the function to use to compute the model predictions.

    # Use the y (from the dataset.data) and the prediction to cumpute the accuracy.

    # Below is a placeholder that allows pytest to pass.

    # If this takes too many resources (e.g., runs very long or causes a memory error), feel free to limit the dataset to the first 10,000 examples.

    accuracy_value = 0.99

    current_time = datetime.now()
    return [Measure(name="accuracy", score=accuracy_value, time=current_time)]
