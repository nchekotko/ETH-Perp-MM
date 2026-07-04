# Markout (adverse-selection) analysis

markout_h = side_sign * (mid(t_fill + h) - fill_price); raw edge = side_sign * (mid(t_fill) - fill_price)  [USD per unit, side_sign = +1 buys / -1 sells]. ret_h = mean(markout_h) / mean(edge), shown when mean(edge) > 0.

| day | n_fills | mean_edge | mean_mo_1s | med_mo_1s | ret_1s | mean_mo_5s | med_mo_5s | ret_5s | mean_mo_30s | med_mo_30s | ret_30s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-03-19 | 346 | +0.3147 | -0.5673 | -0.3500 | -1.803 | -0.6069 | -0.5500 | -1.928 | -0.8227 | -0.4000 | -2.614 |
| 2026-03-20 | 290 | +0.2809 | -0.4579 | -0.3500 | -1.630 | -0.5541 | -0.3500 | -1.973 | -0.6155 | -0.4000 | -2.192 |
| 2026-03-21 | 210 | +0.2069 | -0.4679 | -0.3000 | -2.261 | -0.5338 | -0.2500 | -2.580 | -0.8390 | -0.5500 | -4.055 |
| TOTAL | 846 | +0.2764 | -0.5051 | -0.3500 | -1.828 | -0.5707 | -0.3500 | -2.065 | -0.7557 | -0.4500 | -2.735 |
