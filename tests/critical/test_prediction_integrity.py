"""Prediction artifacts must never silently align different held-out cohorts."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.utils.prediction_tables import (
    BinaryTestPredictions,
    FinalTestPredictions,
    PredictionTableAccumulator,
    load_prediction_snapshot,
)

FINGERPRINTS = {"mimic": "synthetic-source-a", "tudd": "synthetic-source-b", "retriever": "synthetic-query"}


def _predictions(offset=0):
    # Same positional IDs in two sources must not imply the same sample identity.
    return FinalTestPredictions(
        mimic=BinaryTestPredictions(np.array([8, 2, 19]), np.array([0, 1, 0]), np.array([.2, .8, .4]) + offset),
        tudd=BinaryTestPredictions(np.array([8, 2, 19]), np.array([1, 0, 0]), np.array([.6, .3, .7]) + offset),
    )


def test_accumulation_roundtrip_preserves_source_identity_labels_and_model_columns(tmp_path):
    """A wider snapshot preserves earlier predictions and supported v1 files remain readable."""
    accumulator = PredictionTableAccumulator(FINGERPRINTS)
    accumulator.add("tree__0", _predictions())
    accumulator.write(tmp_path)
    first = load_prediction_snapshot(tmp_path)
    accumulator.add("tree__1", _predictions(.05))
    accumulator.write(tmp_path)
    wider = load_prediction_snapshot(tmp_path)
    assert first.generation_id != wider.generation_id
    assert wider.model_instance_ids == ("tree__0", "tree__1")
    for cohort, table in wider.tables.items():
        pd.testing.assert_frame_equal(table[first.tables[cohort].columns], first.tables[cohort])
        original = getattr(_predictions(), cohort)
        np.testing.assert_array_equal(table.y_true, original.y_true)
        np.testing.assert_allclose(table.y_pred_tree__0, original.positive_class_probability)
        np.testing.assert_allclose(table.y_pred_tree__1, original.positive_class_probability + .05)
        assert table.test_set_id.is_unique
        assert table.test_set_id.str.fullmatch(r"ts_[0-9a-f]{64}").all()
    assert set(wider.tables["mimic"].test_set_id).isdisjoint(wider.tables["tudd"].test_set_id)

    changed_source = PredictionTableAccumulator({**FINGERPRINTS, "mimic": "different-file-version"})
    changed_source.add("tree__0", _predictions())
    assert set(first.tables["mimic"].test_set_id).isdisjoint(changed_source.frames()["mimic"].test_set_id)
    # CSV decoding promotes int8 labels to int64; values and row identity are the contract.
    pd.testing.assert_frame_equal(first.tables["tudd"], changed_source.frames()["tudd"], check_dtype=False)

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = "1"
    manifest.pop("datasets")
    manifest_path.write_text(json.dumps(manifest))
    legacy = load_prediction_snapshot(tmp_path)
    assert legacy.generation_id == wider.generation_id
    for cohort in ("mimic", "tudd"):
        pd.testing.assert_frame_equal(legacy.tables[cohort], wider.tables[cohort])


@pytest.mark.parametrize("change", ["identity", "order", "labels", "cohorts"])
def test_changed_second_cohort_is_rejected_atomically_and_next_model_can_accumulate(change):
    """Reject even a late-cohort mismatch without partially adding the new model to MIMIC."""
    accumulator = PredictionTableAccumulator(FINGERPRINTS)
    accumulator.add("first", _predictions())
    before = accumulator.frames()
    incoming = _predictions(.05)
    tudd = incoming.tudd
    if change == "identity":
        incoming = replace(incoming, tudd=replace(tudd, test_set_id=np.array([8, 2, 999])))
    elif change == "order":
        incoming = replace(incoming, tudd=BinaryTestPredictions(
            tudd.test_set_id[::-1], tudd.y_true[::-1], tudd.positive_class_probability[::-1],
        ))
    elif change == "labels":
        incoming = replace(incoming, tudd=replace(tudd, y_true=np.array([0, 1, 0])))
    else:
        incoming = replace(incoming, retriever=tudd)
    with pytest.raises(ValueError, match="test sets? changed"):
        accumulator.add("rejected", incoming)
    assert accumulator.model_instance_ids == ("first",)
    for cohort, table in before.items():
        pd.testing.assert_frame_equal(accumulator.frames()[cohort], table)
    accumulator.add("later", _predictions(.05))
    assert accumulator.model_instance_ids == ("first", "later")
    for cohort, table in accumulator.frames().items():
        np.testing.assert_allclose(table.y_pred_later, getattr(_predictions(.05), cohort).positive_class_probability)


def test_torn_snapshot_rejects_mixed_generations_instead_of_loading_partial_models(tmp_path):
    """Simulate interrupted upload: one old cohort CSV alongside the new manifest."""
    accumulator = PredictionTableAccumulator(FINGERPRINTS)
    accumulator.add("first", _predictions())
    accumulator.write(tmp_path)
    first_csv = (tmp_path / "mimic.csv").read_bytes()
    accumulator.add("later", _predictions(.05))
    accumulator.write(tmp_path)
    (tmp_path / "mimic.csv").write_bytes(first_csv)
    with pytest.raises(ValueError, match="hash mismatch for mimic"):
        load_prediction_snapshot(tmp_path)


def test_source_ids_must_be_positions_not_patient_identifiers():
    """Keep identity generation tied to source-file row positions, never clinical identifiers."""
    predictions = _predictions()
    invalid = replace(predictions.tudd, test_set_id=np.array(["synthetic-a", "synthetic-b", "synthetic-c"]))
    accumulator = PredictionTableAccumulator(FINGERPRINTS)
    with pytest.raises(TypeError, match="integer source-row positions"):
        accumulator.add("invalid", replace(predictions, tudd=invalid))
    assert not accumulator
