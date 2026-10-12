# V30 Results

| Fold 0 candidate | Raw | Direct pair-aware |
|---|---:|---:|
| Reference (`lr=2e-4`) | 493/652 | 550/652 |
| `epochs=1` | 488/652 | 548/652 |
| `epochs=3` | **498/652** | 545/652 |
| confidence `0.4` | 492/652 | 547/652 |
| confidence `0.5` | 492/652 | 548/652 |
| confidence `0.7` | 492/652 | 550/652 |
| confidence `0.8` | 492/652 | **551/652** |
| `lr=1e-4` | 490/652 | 548/652 |
| `lr=3e-4` | 497/652 | **551/652** |
| retention `0.5` | 494/652 | **551/652** |

The predeclared tie-break selects `lr=3e-4`. Blind Fold 1 scores `616/705`,
giving `1167/1357`, one row below the `lr=2e-4` reference. More aggressive
head adaptation does not generalize and is rejected.
