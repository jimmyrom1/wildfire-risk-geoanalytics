from ml.train_model import run


def test_training_pipeline_runs_and_returns_valid_metrics():
    results = run()

    clf_metrics = results["clasificador_ocurrencia"]
    assert 0 <= clf_metrics["test_accuracy"] <= 1
    assert 0 <= clf_metrics["test_f1"] <= 1
    assert 0 <= clf_metrics["test_roc_auc"] <= 1

    reg_metrics = results["regresor_severidad"]
    assert reg_metrics["test_mae_log_area"] >= 0
    assert reg_metrics["test_mae_hectareas"] >= 0
