import argparse
from funtions import run_experiment, create_experiment, register_best_model, promote_model_to_stage, call_model_at_stage


if __name__ == '__main__':
    # mlflow.set_tracking_uri("http://myserver.com/mlflow:5000")
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--nsplits', type=int, default=5)
    # parser.add_argument('--nephocs', type=int, default=500)
    # args = parser.parse_args()

    # Create a new experiment in MLflow and get experiment ID
    experiment_name = f"Web Classifier"
    experiment_id = create_experiment(experiment_name)

    # run_experiment(experiment_id, n_splits=args.nsplits)

    model_name = "Web-classifier-best-model"
    stage = "Production"
    register_best_model(experiment_id, "accuracy", model_name)
    promote_model_to_stage(model_name, stage)

    # data = [{
    #     "sepal-length": 6.9,
    #     "sepal-width": 3.1,
    #     "petal-length": 5.1,
    #     "petal-width": 2.3
    # }]
    # predictions = call_model_at_stage(model_name, stage, data)
    # print(predictions)

