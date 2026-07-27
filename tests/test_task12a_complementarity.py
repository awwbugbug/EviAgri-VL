import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).parents[1]; sys.path.insert(0,str(ROOT/"server"))
from build_task12a_local_dataset import select_rows
from evaluate_task12a_complementarity import bootstrap, branch


def test_fresh_selection_has_frozen_counts_and_excludes_prior_ids():
    base=[]
    for split,count in (("train",5),("val",2),("dev",4)):
        for class_id in range(16):
            for i in range(count): base.append({"id":f"{split}-{class_id}-{i}","split":split,"class_id":class_id,"near_duplicate_component_id":f"{split}-{class_id}-{i}","image":"x"})
    nulls=[{"id":f"n-{i}","mask_ratio":i/1000,"image":"x","mask":"m"} for i in range(100)]
    used={"dev-0-0","n-0"}
    rows=select_rows(base,nulls,used)
    counts={s:sum(r["probe_split"]==s for r in rows) for s in ("probe_train","probe_val","probe_test","null_test")}
    assert counts=={"probe_train":64,"probe_val":16,"probe_test":32,"null_test":32}
    assert not ({r["id"] for r in rows}&used)


def test_branch_prioritizes_h1_only_for_safe_conditional_gain():
    assert branch({"accuracy":1/32,"true_probability":.01,"null_confidence":0.,"confidence_auroc":.01})["priority"]=="H1_PRIORITY"
    assert branch({"accuracy":1/32,"true_probability":.01,"null_confidence":.01,"confidence_auroc":-.01})["priority"]=="H3_PRIORITY"
    result=branch({"accuracy":0.,"true_probability":.01,"null_confidence":0.,"confidence_auroc":0.})
    assert result["priority"]=="H2_PRIORITY" and not result["authorize_training"]


def test_bootstrap_reports_fresh_image_unit():
    result=bootstrap(lambda p,n:float(p.mean()-n.mean()),4,4,100,9)
    assert result["unit"]=="fresh_image" and result["repetitions"]==100


def test_shell_uses_one_protocol_attempt_and_no_large_training():
    text=(ROOT/"server"/"run_task12a_conditional_complementarity.sh").read_text(encoding="utf-8")
    assert "task12a_conditional_complementarity/2026-07-27/protocol_v1/attempt_01" in text
    assert "--repetitions 1000" in text and "local_features" in text
    for forbidden in ("shutdown","poweroff","Task8","7B","SAM2","qlora"):
        assert forbidden not in text
