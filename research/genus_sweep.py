"""genus_sweep.py -- R7: genus of the sealed surface of spectreseal.py over thickness, blend radius and depth.

    python research/genus_sweep.py OUT.json [depths ...]
Each row records the spectreseal analyse() report (components, genus, Euler, sealed) for one setting; a genus that
is the same over a range of blend at fixed depth is a plateau, otherwise it is a render artefact."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import spectreseal as SS  # noqa: E402

THICK = (0.20, 0.30, 0.40)
BLEND = (1.5, 3.0, 5.0, 8.0)


def main():
    out = sys.argv[1]
    depths = [int(x) for x in sys.argv[2:]] or [2, 3, 4]
    rows = []
    for d in depths:
        for t in THICK:
            for b in BLEND:
                try:
                    _, _, _, info, _ = SS.build(d, "+1", vox=min(1.5, b / 2), thick=t, blend=b)
                    row = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in info.items()}
                except Exception as e:  # record and continue
                    row = dict(depth=d, error=repr(e))
                row.update(thick=t, blend=b)
                print(json.dumps(row), flush=True)
                rows.append(row)
                with open(out, "w") as f:
                    json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
