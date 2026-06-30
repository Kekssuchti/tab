# Test inference configs

All tested in extreme case: 
- train on MIMIC + 2500 samples from tudd
- test on both testsets
- code in [single_model_training.py](./single_model_training.py)

## TABPFN3

MODEL_PARAMS = {
    "random_state": RANDOM_STATE,
    "predict_batch_size": 2048,
    "fit_mode": "fit_with_cache",
    "inference_config": {"SUBSAMPLE_SAMPLES": 15_000},
}

MODEL_PARAMS = {
    "random_state": RANDOM_STATE,
    "predict_batch_size": 4096,
    "fit_mode": "fit_with_cache",
    "inference_config": {"SUBSAMPLE_SAMPLES": 15_000},
}

(moved most parameters to default)

MODEL_PARAMS =  {
  "n_estimators": 8
}
(4.2GB peak ram usage)



#### Training

| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabpfn-3 | 29.716628048 | false | 0.9445928475 | 0.6225040983 | 0.286067805 | 0.938560811 | 0.1716152019 | 0.8588410104 | 2 |
| tabpfn-3 | 29.650290496 | false | 0.9442474025 | 0.6211384357 | 0.2868447082 | 0.938582107 | 0.1722090261 | 0.8579881657 | 2 |

#### Testing

| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 18.898126706 | 588.1535335706 | 0.9026417448 | 0.5112247106 | 0.2531120332 | 0.9352226721 | 0.1513647643 | 0.7721518987 | 2 |
| tudd | 4068 | 1 | 6.913492495 | 588.4146114198 | 0.8926781228 | 0.4478854252 | 0.2197802198 | 0.947640118 | 0.1287553648 | 0.75 | 2 |
| mimic | 11115 | 1 | 20.936159723 | 530.8996562435 | 0.9025490748 | 0.5104745688 | 0.2531120332 | 0.9352226721 | 0.1513647643 | 0.7721518987 | 2 |
| tudd | 4068 | 1 | 7.236060548 | 562.1843505891 | 0.8924554168 | 0.4452499858 | 0.2132352941 | 0.947394297 | 0.1244635193 | 0.7435897436 | 2 |
| mimic | 11115 | 1 | 11.45384542 | 970.4164490111 | 0.9051944421 | 0.5277565244 | 0.3711151737 | 0.9381016644 | 0.2518610422 | 0.7048611111 | 2 |
| tudd | 4068 | 1 | 3.078071519 | 1321.6067186509 | 0.8988668856 | 0.4977633037 | 0.3821656051 | 0.9523107178 | 0.2575107296 | 0.7407407407 | 2 |


## TabPFN-2.5

MODEL_PARAMS = {
    "n_estimators": 1,
}
(12GB Peak Ram usage)

MODEL_PARAMS = {
    "n_estimators": 2,
}
(20GB peak ram usage)

-> fix
MODEL_PARAMS = {
    "n_estimators": 8,
    "inference_config": {"SUBSAMPLE_SAMPLES": 15_000},
    "predict_batch_size": 2048, #!
}
-> with 4096 it also worked(and was a bit faster), 8192 could get close to ram limit.


| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabpfn-2.5 | 18.960183849 | false | 0.9344746114 | 0.6732908156 | 0.4244080146 | 0.9461635113 | 0.2767220903 | 0.91015625 | 2 |




| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 15.774530678 | 704.6168426108 | 0.8946759162 | 0.4905392237 | 0.3143683703 | 0.9360323887 | 0.2022332506 | 0.7056277056 | 2 |
| tudd | 4068 | 1 | 5.412442234 | 751.6015551069 | 0.8654928908 | 0.4409835386 | 0.2765957447 | 0.9498525074 | 0.1673819742 | 0.7959183673 | 2 |
| mimic | 11115 | 1 | 28.515464347 | 389.7884973832 | 0.8951152562 | 0.5040136784 | 0.3396946565 | 0.9377417904 | 0.2208436725 | 0.7355371901 | 2 |
| tudd | 4068 | 1 | 9.733250812 | 417.9487489406 | 0.8741174298 | 0.4429087139 | 0.3219178082 | 0.9513274336 | 0.2017167382 | 0.7966101695 | 2 |
| mimic | 11115 | 1 | 54.554098189 | 203.7427135445 | 0.8966961221 | 0.5007049157 | 0.3191489362 | 0.9366621682 | 0.2047146402 | 0.7236842105 | 2 |
| tudd | 4068 | 1 | 17.089245867 | 238.0444421983 | 0.869018695 | 0.424474373 | 0.2857142857 | 0.9496066863 | 0.1759656652 | 0.7592592593 | 2 |


----


## TabPFN-2.6
MODEL_PARAMS = {
    "n_estimators": 8,
    "fit_mode": "fit_preprocessors",
}
-> Doesn not have fit_cache -> really slow (needs to refit for every prediction, only preprocessors are kept in cache)

| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 763.825272641 | 14.5517573169 | 0.9013268538 | 0.5080999848 | 0.3386944182 | 0.9371120108 | 0.2220843672 | 0.7131474104 | 2 |
| tudd | 4068 | 1 | 255.645095386 | 15.9126854902 | 0.8974690982 | 0.491120103 | 0.3540983607 | 0.9515732547 | 0.2317596567 | 0.75 | 2 |

## TabICL-2
MODEL_PARAMS = {
    "random_state": RANDOM_STATE,
    "feat_shuffle_method": "latin",
    "class_shuffle_method": "shift",
    "support_many_classes": False,
    "batch_size": 1,
    "kv_cache": "repr",
    "predict_batch_size": 8192,
}

MODEL_PARAMS = {
    "random_state": RANDOM_STATE,
    "feat_shuffle_method": "latin",
    "class_shuffle_method": "shift",
    "support_many_classes": False,
    "batch_size": 1,
    "kv_cache": "kv",
    "predict_batch_size": 8192,
}
MODEL_PARAMS = {
    "random_state": RANDOM_STATE,
    "feat_shuffle_method": "latin",
    "class_shuffle_method": "shift",
    "support_many_classes": False,
    "batch_size": 1,
    "kv_cache": "kv",
} (no batch aka all predictions at once)
MUST ENABLE FA!!!
Native Batch-Size == Ensambe member not rows
(avg. vram 11GB or so but peeks still full 24GB)

| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabicl-2 | 13.439223595 | false | 0.9999368221 | 0.9992568668 | 0.9470229723 | 0.9927806291 | 0.8996437055 | 0.9996700759 | 2 |
| tabicl-2 | 25.161887905 | false | 0.9999367438 | 0.999256264 | 0.9468583932 | 0.992759333 | 0.8993467933 | 0.999669967 | 2 |
| tabicl-2 | 25.141082822 | false | 0.9999369311 | 0.9992579158 | 0.9473519763 | 0.9928232212 | 0.9002375297 | 0.9996702934 | 2 |



| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 27.689907712 | 401.4097885629 | 0.902690908 | 0.5189469464 | 0.3761301989 | 0.9379217274 | 0.2580645161 | 0.6933333333 | 2 |
| tudd | 4068 | 1 | 13.368744443 | 304.2918515905 | 0.9022175468 | 0.5144564317 | 0.4373177843 | 0.9525565388 | 0.321888412 | 0.6818181818 | 2 |
| mimic | 11115 | 1 | 3.801398164 | 2923.924177494 | 0.9026933752 | 0.5188708973 | 0.3764705882 | 0.9380116959 | 0.2580645161 | 0.6956521739 | 2 |
| tudd | 4068 | 1 | 1.467920096 | 2771.2680077616 | 0.9021990812 | 0.5144590267 | 0.4373177843 | 0.9525565388 | 0.321888412 | 0.6818181818 | 2 |
| mimic | 11115 | 1 | 3.690195899 | 3012.0352155334 | 0.9026815808 | 0.5188744552 | 0.3764705882 | 0.9380116959 | 0.2580645161 | 0.6956521739 | 2 |
| tudd | 4068 | 1 | 1.468890148 | 2769.4378681332 | 0.9021990812 | 0.5144590267 | 0.4373177843 | 0.9525565388 | 0.321888412 | 0.6818181818 | 2 |


----
## Mitra
MODEL_PARAMS = {
    "random_state": RANDOM_STATE,
}
(9GB Vram)
MODEL_PARAMS = {
  "n_estimators": 4,
}

| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| mitra | 28.016096127 | false | 0.8771259027 | 0.4235237717 | 0.2345132743 | 0.9336840088 | 0.1416270784 | 0.6814285714 | 2 |
| mitra | 109.122007481 | false | 0.8861916229 | 0.4433462601 | 0.2434484448 | 0.9342164108 | 0.1475653207 | 0.6951048951 | 2 |

| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 29.199188758 | 380.6612605617 | 0.880619322 | 0.4377458848 | 0.2436548223 | 0.9329734593 | 0.1488833747 | 0.6703910615 | 2 |
| tudd | 4068 | 1 | 10.663507709 | 381.4879785351 | 0.8288577648 | 0.3378032126 | 0.1264822134 | 0.9456735497 | 0.0686695279 | 0.8 | 2 |
| mimic | 11115 | 1 | 116.251890587 | 95.6113482876 | 0.8870721023 | 0.4499784374 | 0.2510121457 | 0.9334233018 | 0.1538461538 | 0.6813186813 | 2 |
| tudd | 4068 | 1 | 42.504564587 | 95.7073678916 | 0.8531528557 | 0.3677475989 | 0.112 | 0.9454277286 | 0.0600858369 | 0.8235294118 | 2 |



----
## Limix
2M

MODEL_NAME = "limix-2m"
MODEL_PARAMS = {
    "random_state": RANDOM_STATE,
    "predict_batch_size": 16384
}

remove training predictions
MODEL_PARAMS = {
    "random_state": RANDOM_STATE,
}


| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| limix-2m | 0.012098377 | false | 0.8644807973 | 0.3954504788 | 0.3904089145 | 0.8695189216 | 0.5825415677 | 0.2935807272 | 2 |


| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 234.072597391 | 47.4852679207 | 0.7164606224 | 0.2141196212 | 0.245243129 | 0.8394062078 | 0.3598014888 | 0.1860166774 | 2 |
| tudd | 4068 | 1 | 156.062347277 | 26.0665052845 | 0.8704164825 | 0.4196726513 | 0.3953488372 | 0.9488692232 | 0.2918454936 | 0.6126126126 | 2 |
| mimic | 11115 | 1 | 240.835817866 | 46.1517730149 | 0.7164606224 | 0.2141196212 | 0.245243129 | 0.8394062078 | 0.3598014888 | 0.1860166774 | 2 |
| tudd | 4068 | 1 | 154.541512598 | 26.3230243552 | 0.8704164825 | 0.4196726513 | 0.3953488372 | 0.9488692232 | 0.2918454936 | 0.6126126126 | 2 |


16M

| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 340.732891067 | 32.6208601852 | 0.901966698 | 0.5000230588 | 0.332688588 | 0.9379217274 | 0.2133995037 | 0.7543859649 | 2 |
| tudd | 4068 | 1 | 154.068592772 | 26.4038239515 | 0.9037149364 | 0.5027471736 | 0.3766233766 | 0.9528023599 | 0.2489270386 | 0.7733333333 | 2 |


---
## Orion BIX

MODEL_PARAMS = {
    "random_state": RANDOM_STATE,
    "n_estimators": 8,
}


| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 394.560187329 | 28.1706070631 | 0.8601302868 | 0.4164112793 | 0.0738095238 | 0.9300044984 | 0.0384615385 | 0.9117647059 | 2 |
| tudd | 4068 | 1 | 346.926187756 | 11.7258372056 | 0.8378958206 | 0.3623785277 | 0.0901639344 | 0.9454277286 | 0.0472103004 | 1 | 2 |
