import numpy as np
import pandas as pd
import pytest

from src.utils.prediction_metrics import recompute_classification_metrics
from src.utils.prediction_tables import (
    PREDICTION_MANIFEST_FILENAME,
    BinaryTestPredictions,
    FinalTestPredictions,
    PredictionTableAccumulator,
    load_prediction_snapshot,
)

_COHORT_FINGERPRINTS = {
    "mimic": "sha256:" + "a" * 64,
    "tudd": "sha256:" + "b" * 64,
}


def _accumulator() -> PredictionTableAccumulator:
    return PredictionTableAccumulator(_COHORT_FINGERPRINTS)


def _predictions(
    mimic_probability=(0.1, 0.2, 0.8, 0.9),
    tudd_probability=(0.1, 0.6, 0.4, 0.9),
) -> FinalTestPredictions:
    return FinalTestPredictions(
        mimic=BinaryTestPredictions(
            test_set_id=np.array([101, 102, 103, 104]),
            y_true=np.array([0, 0, 1, 1]),
            positive_class_probability=np.array(mimic_probability),
        ),
        tudd=BinaryTestPredictions(
            test_set_id=np.array([201, 202, 203, 204]),
            y_true=np.array([0, 0, 1, 1]),
            positive_class_probability=np.array(tudd_probability),
        ),
    )


def test_prediction_snapshot_accumulates_wider_generations_and_validates_hashes(tmp_path):
    accumulator = _accumulator()
    accumulator.add("model", _predictions())
    accumulator.write_csvs(tmp_path)
    first = load_prediction_snapshot(tmp_path)

    accumulator.add(
        "model__1",
        _predictions(
            mimic_probability=(0.2, 0.3, 0.7, 0.8),
            tudd_probability=(0.2, 0.5, 0.5, 0.8),
        ),
    )
    paths = accumulator.write_csvs(tmp_path)
    second = load_prediction_snapshot(tmp_path)

    assert accumulator.model_instance_ids == ("model", "model__1")
    assert {path.name for path in paths} == {"mimic.csv", "tudd.csv"}
    assert (tmp_path / PREDICTION_MANIFEST_FILENAME).exists()
    assert first.generation_id != second.generation_id
    assert second.model_instance_ids == ("model", "model__1")
    mimic = second.tables["mimic"]
    assert list(mimic.columns) == ["test_set_id", "y_true", "y_pred_model", "y_pred_model__1"]
    assert mimic["y_true"].tolist() == [0, 0, 1, 1]
    assert mimic["test_set_id"].str.fullmatch(r"ts_[0-9a-f]{64}").all()
    assert not set(mimic["test_set_id"]).intersection({"101", "102", "103", "104"})
    np.testing.assert_allclose(mimic["y_pred_model"], [0.1, 0.2, 0.8, 0.9])


def test_stable_test_ids_depend_on_source_file_fingerprint():
    first = _accumulator()
    first.add("model", _predictions())
    same_cohort = _accumulator()
    same_cohort.add("model", _predictions())
    changed_cohort = PredictionTableAccumulator({"mimic": "sha256:" + "c" * 64, "tudd": _COHORT_FINGERPRINTS["tudd"]})
    changed_cohort.add("model", _predictions())

    first_ids = first.frames()["mimic"]["test_set_id"]
    assert first_ids.equals(same_cohort.frames()["mimic"]["test_set_id"])
    assert not first_ids.equals(changed_cohort.frames()["mimic"]["test_set_id"])


def test_prediction_tables_reject_nonpositional_source_ids():
    predictions = _predictions()
    patient_like_ids = BinaryTestPredictions(
        test_set_id=np.array(["patient-a", "patient-b", "patient-c", "patient-d"]),
        y_true=predictions.mimic.y_true,
        positive_class_probability=predictions.mimic.positive_class_probability,
    )

    with pytest.raises(TypeError, match="integer row positions"):
        _accumulator().add(
            "model",
            FinalTestPredictions(mimic=patient_like_ids, tudd=predictions.tudd),
        )


def test_prediction_snapshot_rejects_torn_csv_generation(tmp_path):
    accumulator = _accumulator()
    accumulator.add("model", _predictions())
    accumulator.write_csvs(tmp_path)
    mimic_path = tmp_path / "mimic.csv"
    mimic_path.write_text(mimic_path.read_text(encoding="utf-8") + chr(10), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_prediction_snapshot(tmp_path)


def test_prediction_table_add_is_atomic_when_test_identity_changes():
    accumulator = _accumulator()
    accumulator.add("model", _predictions())
    mismatched = _predictions()
    mismatched_tudd = BinaryTestPredictions(
        test_set_id=np.array([201, 202, 203, 999]),
        y_true=mismatched.tudd.y_true,
        positive_class_probability=mismatched.tudd.positive_class_probability,
    )

    with pytest.raises(ValueError, match="IDs or labels changed"):
        accumulator.add(
            "other",
            FinalTestPredictions(mimic=mismatched.mimic, tudd=mismatched_tudd),
        )

    assert accumulator.model_instance_ids == ("model",)
    assert "y_pred_other" not in accumulator.frames()["mimic"]


def test_recomputed_metrics_include_current_scores_confidence_intervals_and_delta():
    accumulator = _accumulator()
    accumulator.add("model", _predictions())

    metrics = recompute_classification_metrics(
        accumulator.frames(),
        n_bootstrap=50,
        random_state=7,
    )

    assert metrics[["scope", "dataset"]].to_records(index=False).tolist() == [
        ("test", "mimic"),
        ("test", "tudd"),
        ("test_delta", "mimic_minus_tudd"),
    ]
    mimic = metrics.loc[metrics["dataset"].eq("mimic")].iloc[0]
    tudd = metrics.loc[metrics["dataset"].eq("tudd")].iloc[0]
    difference = metrics.loc[metrics["scope"].eq("test_delta")].iloc[0]

    assert mimic["roc_auc"] == pytest.approx(1.0)
    assert mimic["prc_auc"] == pytest.approx(1.0)
    assert mimic["accuracy"] == pytest.approx(1.0)
    assert tudd["accuracy"] == pytest.approx(0.5)
    for metric in ("roc_auc", "prc_auc", "f1", "accuracy", "sensitivity", "precision"):
        assert mimic[f"{metric}_ci_lower"] <= mimic[f"{metric}_ci_upper"]
        assert difference[metric] == pytest.approx(mimic[metric] - tudd[metric])
    assert pd.isna(difference["roc_auc_ci_lower"])
