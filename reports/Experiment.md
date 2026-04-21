# Experiment

## Baseline MLP

### Fequency features

Apply only `src/preprocessing/de_feat_cal.py` to extract features to train a model

#### Train and test on the same subject

| Experiment | Rev | Subject | Metric | Model | Feature | Top1 | Top5 | Epoch |
|---|---|---:|---|---|---|---:|---:|---:|
| frank-sick | a8223e8 | 0 | wt | mlp | freq | 0.5337 | 0.8125 | 733 |
| busty-snob | a8223e8 | 1 | wt | mlp | freq | 0.5249 | 0.8125 | 733 |
| eight-cham | bc1f8c0 | 2 | wt | mlp | freq | 0.5275 | 0.9287 | 734 |
| ratty-food | 74c267d | 3 | wt | mlp | freq | 0.5713 | 0.9038 | 493 |
| goosy-snob | 10e9344 | 4 | wt | mlp | freq | 0.3013 | 0.7412 | 481 |
| busty-rugs | 1977ba9 | 5 | wt | mlp | freq | 0.5175 | 0.8425 | 391 |
| dirty-muss | e0426bb | 6 | wt | mlp | freq | 0.5663 | 0.9537 | 798 |
| jammy-yoni | bcecf02 | 7 | wt | mlp | freq | 0.4275 | 0.7650 | 756 |
| tamer-math | e0cef85 | 8 | wt | mlp | freq | 0.6863 | 0.8625 | 822 |
| pearl-pita | 467cc33 | 9 | wt | mlp | freq | 0.6412 | 0.8688 | 656 |
| drear-doek | cae50c2 | 10 | wt | mlp | freq | 0.4738 | 0.7825 | 634 |
| butch-peel | 132db2b | 11 | wt | mlp | freq | 0.6800 | 0.9587 | 688 |
| weepy-main | 57b6555 | 12 | wt | mlp | freq | 0.3275 | 0.7312 | 678 |
| metal-yoke | 5fae285 | 13 | wt | mlp | freq | 0.5913 | 0.8888 | 814 |
| moral-agio | 8afb66a | 14 | wt | mlp | freq | 0.5175 | 0.8037 | 647 |
| hulky-deys | 49631d3 | 15 | wt | mlp | freq | 0.5188 | 0.9163 | 699 |

Average Top1: 0.5254
Average Top5: 0.8483

#### Train and test with all subjects

| Experiment | Rev | Subject | Metric | Model | Feature | Top1 | Top5 | Epoch |
|---|---|---:|---|---|---|---:|---:|---:|
| frank-sick | a8223e8 | -1 | wt | mlp | freq | 0.1673 | 0.4559 | 219 |
| frank-sick | a8223e8 | -1 | ct | mlp | freq | 0.0354 | 0.1171 | 787 |

### Time features

Using raw time-series data as input to train a MLP

| Experiment | Rev | Subject | Metric | Model | Feature | Top1 | Top5 | Epoch |
|---|---|---:|---|---|---|---:|---:|---:|
| alpha-rale | ce52e87 | 0 | wt | mlp | time | 0.0600 | 0.2000 | 845 |
| ionic-prof | fc6eee4 | 1 | wt | mlp | time | 0.1000 | 0.2700 | 490 |
| manky-joss | 3840276 | 2 | wt | mlp | time | 0.0512 | 0.1963 | 820 |
| mirky-rick | 3f219bb | 3 | wt | mlp | time | 0.0900 | 0.2462 | 742 |
| forty-huts | d0f7d68 | 4 | wt | mlp | time | 0.0725 | 0.2200 | 814 |
| adunc-doge | e9dc508 | 5 | wt | mlp | time | 0.0625 | 0.2238 | 995 |
| farci-coed | 220d0d2 | 6 | wt | mlp | time | 0.1013 | 0.2988 | 945 |
| genic-luce | 659e1f3 | 7 | wt | mlp | time | 0.0625 | 0.2050 | 866 |
| soled-cool | 54b3757 | 8 | wt | mlp | time | 0.1013 | 0.2750 | 197 |
| prosy-prow | d200a57 | 9 | wt | mlp | time | 0.0700 | 0.2238 | 405 |
| enemy-maya | c3e5a25 | 10 | wt | mlp | time | 0.0788 | 0.2487 | 991 |
| olden-merk | 0d3c22c | 11 | wt | mlp | time | 0.0775 | 0.2062 | 714 |
| kinky-hate | c978aab | 12 | wt | mlp | time | 0.0900 | 0.3075 | 797 |
| tonic-wort | 50f0e85 | 13 | wt | mlp | time | 0.0575 | 0.1850 | 985 |
| woozy-erns | 8ae0fa7 | 14 | wt | mlp | time | 0.0700 | 0.1913 | 818 |
| blond-linn | 8281594 | 15 | wt | mlp | time | 0.1025 | 0.2625 | 751 |

Average Top1: 0.0780
Average Top5: 0.2350

#### Train and test with all subjects

| Experiment | Rev | Subject | Metric | Model | Feature | Top1 | Top5 | Epoch |
|---|---|---:|---|---|---|---:|---:|---:|
| frank-sick | a8223e8 | -1 | wt | mlp | time | 0.0522 | 0.2023 | 362 |
| frank-sick | a8223e8 | -1 | ct | mlp | time | 0.03983 | 0.1611 | 455 |