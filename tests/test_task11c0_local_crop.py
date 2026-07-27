import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/"server"))
from build_task11c0_local_crop_smoke import expand_box, largest_component_box


def test_expand_box_clamps_and_contains_evidence():
    assert expand_box((2,3,8,9),10,12,.5)==(0,0,10,12)


def test_largest_component_uses_eight_connectivity_and_not_union_box():
    mask=np.zeros((10,12),dtype=bool);mask[1:4,1:4]=1;mask[7:9,9:11]=1
    box,size=largest_component_box(mask)
    assert box==(1,1,4,4) and size==9


def test_shell_uses_protocol_attempt_and_never_trains_or_shuts_down():
    text=(ROOT/"server"/"run_task11c0_local_crop_smoke.sh").read_text(encoding="utf-8")
    assert "task11c0_local_crop/2026-07-27/protocol_v1/attempt_01" in text
    for forbidden in ("shutdown","poweroff","train","Task8"):
        assert forbidden not in text
