# Markout (adverse-selection) analysis

markout_h = side_sign * (mid(t_fill + h) - fill_price); raw edge = side_sign * (mid(t_fill) - fill_price)  [USD per unit, side_sign = +1 buys / -1 sells]. ret_h = mean(markout_h) / mean(edge), shown when mean(edge) > 0.

| day | n_fills | mean_edge | mean_mo_1s | med_mo_1s | ret_1s | mean_mo_5s | med_mo_5s | ret_5s | mean_mo_30s | med_mo_30s | ret_30s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-03-19 | 24 | +0.2750 | -0.6583 | -0.5000 | -2.394 | -0.9083 | -0.9500 | -3.303 | -0.8500 | -0.8500 | -3.091 |
| 2026-03-20 | 22 | +0.3477 | -0.4045 | -0.1000 | -1.163 | -0.1045 | -0.0500 | -0.301 | -0.3636 | -0.5000 | -1.046 |
| 2026-03-21 | 19 | +0.4184 | -0.3553 | -0.3500 | -0.849 | -0.4132 | -0.1500 | -0.987 | -0.6342 | -0.5500 | -1.516 |
| TOTAL | 65 | +0.3415 | -0.4838 | -0.3500 | -1.417 | -0.4915 | -0.4500 | -1.439 | -0.6223 | -0.5500 | -1.822 |
