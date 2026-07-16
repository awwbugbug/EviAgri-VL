import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from score_task10c_c2_candidates import (
    candidate_targets,
    mean_active_token_logprob,
    rank_candidate_scores,
    resource_preflight_decision,
    validate_candidate_results,
)
from task10c_contract import CLASS_IDS, strict_parse_pest_json


def test_candidate_targets_are_exact_frozen_json():
    targets = candidate_targets()
    assert targets[0] == '{"pest_id":"IP009"}'
    assert targets[-1] == '{"pest_id":"IP101"}'
    assert len(targets) == 16
    assert len(set(targets)) == 16
    assert all(strict_parse_pest_json(value)["schema_valid"] for value in targets)


def test_active_token_mean_excludes_prompt_and_padding():
    token_log_probs = torch.tensor([
        [-9.0, -9.0, -0.2, -0.4, -9.0],
        [-9.0, -0.1, -0.3, -0.5, -0.7],
    ])
    active = torch.tensor([
        [False, False, True, True, False],
        [False, True, True, False, False],
    ])
    result = mean_active_token_logprob(token_log_probs, active)
    assert result.tolist() == pytest.approx([-0.3, -0.2])
    with pytest.raises(ValueError, match="no active answer tokens"):
        mean_active_token_logprob(token_log_probs[:1], torch.zeros((1, 5), dtype=torch.bool))


def test_rank_reports_top_1_3_5_without_generation():
    scores = {f"IP{x:03d}": -10.0 - index for index, x in enumerate(CLASS_IDS)}
    scores.update({"IP009": -0.1, "IP010": -0.2, "IP016": -0.3,
                   "IP017": -0.4, "IP022": -0.5})
    ranked = rank_candidate_scores(scores, truth="IP016")
    assert ranked["prediction"] == "IP009"
    assert ranked["truth_rank"] == 3
    assert ranked["top1_correct"] is False
    assert ranked["top3_correct"] is True
    assert ranked["top5_correct"] is True
    with pytest.raises(ValueError, match="candidate set"):
        rank_candidate_scores({"IP009": -0.1}, truth="IP009")


def test_resource_preflight_blocks_long_or_nonfinite_projection():
    assert resource_preflight_decision(elapsed_seconds=6.0, rows=160, peak_vram_bytes=10_000)["passed"] is True
    blocked = resource_preflight_decision(elapsed_seconds=12.0, rows=160, peak_vram_bytes=10_000)
    assert blocked["passed"] is False
    assert "30 minutes" in blocked["reason"]
    with pytest.raises(ValueError, match="finite"):
        resource_preflight_decision(elapsed_seconds=float("nan"), rows=160, peak_vram_bytes=10_000)


def test_candidate_results_require_160_unique_source_prompt_rows():
    rows = []
    for source in range(80):
        for prompt in ("train", "unseen"):
            rows.append({
                "id": f"source-{source}::{prompt}",
                "source_image_sha256": f"{source:064x}",
                "prompt_variant": prompt,
                "scores": {f"IP{x:03d}": -float(index) for index, x in enumerate(CLASS_IDS)},
                "truth": f"IP{CLASS_IDS[source % 16]:03d}",
                "truth_rank": source % 16 + 1,
            })
    assert validate_candidate_results(rows)["passed"] is True
    with pytest.raises(ValueError, match="160 unique"):
        validate_candidate_results(rows[:-1])
