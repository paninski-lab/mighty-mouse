Label version of each dataset (`_raw/<dataset>/VERSION.txt`) in each corpus version (`data/head-fixed_vN`). Updated when a full corpus is built (`scripts/preprocess/build_dataset.py`).

| dataset | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 | v10 | v11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aind-vbn-body | – | – | – | – | – | – | – | – | – | 1 | 1 |
| aind-vbn-face | – | – | – | – | – | – | – | – | – | 1 | 1 |
| cavelli | – | – | – | – | – | – | – | – | – | 1 | 1 |
| cazettes-side | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 |
| cheese-2d | h1 | h1 | h1 | h1 | h1 | h2 | h2 | 1 | 1 | 2 | 2 |
| cheese-3d | – | – | h1 | h1 | h2 | h3 | h3 | h3 | h3 | 3 | 3 |
| facemap | h1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| hantman-mv | – | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 4 | 4 |
| ibl | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | h2 | 2 |
| kaufman | – | – | – | – | – | – | 0 | 0 | 1 | 1 | 1 |
| kaufman-water | – | – | – | – | – | – | – | – | – | – | 0 |
| kondo | h1 | h1 | h1 | h1 | h1 | h1 | h1 | 2 | 2 | 2 | 2 |
| tyssowski | – | – | – | – | – | – | – | – | – | – | 1 |

`hN` = a pre-versioning, hash-keyed dataset version with no matching snapshot (e.g. cheese-3d@v2); `?` = no VERSION.txt; `–` = not in that corpus version.
