default_params:




---

## Baseline
MODEL_PARAMS = {
    "n_estimators": 8,
}
(no batchsize)

| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabicl-2 | 1.394902045 | false | 0.9999369073 | 0.9992571184 | 0.9463643471 | 0.9926954448 | 0.898456057 | 0.9996696399 | 2 |

| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 21.486450623 | 517.3027502319 | 0.9026576311 | 0.518645262 | 0.3761301989 | 0.9379217274 | 0.2580645161 | 0.6933333333 | 2 |
| tudd | 4068 | 1 | 19.581245214 | 207.7498113905 | 0.9022897303 | 0.5141893266 | 0.4373177843 | 0.9525565388 | 0.321888412 | 0.6818181818 | 2 |


(batchsize = 2048)

- fit time is small but prediction on train takes a long time. The cell ran for 7min.

| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabicl-2 | 8.819903891 | false | 0.9999350409 | 0.9992376857 | 0.9455399061 | 0.9925889644 | 0.8969714964 | 0.9996690933 | 2 |

| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 110.707569848 | 100.3996385727 | 0.9026640096 | 0.518799686 | 0.3757904246 | 0.9378317589 | 0.2580645161 | 0.6910299003 | 2 |
| tudd | 4068 | 1 | 36.850136387 | 110.3930785297 | 0.9022410484 | 0.5145172367 | 0.4373177843 | 0.9525565388 | 0.321888412 | 0.6818181818 | 2 |


---

MODEL_PARAMS = {
    "n_estimators": 8,
    "kv_cache": "kv"
}

| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabicl-2 | 18.274711565 | false | 0.999938147 | 0.9992724656 | 0.9473519763 | 0.9928232212 | 0.9002375297 | 0.9996702934 | 2 |

| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 3.740678145 | 2971.3863554004 | 0.9026764659 | 0.5186598902 | 0.3775971093 | 0.9380116959 | 0.2593052109 | 0.6943521595 | 2 |
| tudd | 4068 | 1 | 1.38011765 | 2947.5747955254 | 0.9022679074 | 0.5145221205 | 0.4373177843 | 0.9525565388 | 0.321888412 | 0.6818181818 | 2 |



---

MODEL_PARAMS = {
    "n_estimators": 8,
    "kv_cache": "kv"
    "support_many_classes": False
}
- no impact, data identical

---

MODEL_PARAMS = {
    "n_estimators": 8,
    "kv_cache": "kv"
    "use_amp": True
}
-> with False it runs OOM


| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabicl-2 | 25.157644361 | false | 0.9999379665 | 0.9992704737 | 0.9475164011 | 0.9928445173 | 0.9005344418 | 0.9996704021 | 2 |


| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 3.556845403 | 3124.9601094907 | 0.9026785721 | 0.5185995944 | 0.3761301989 | 0.9379217274 | 0.2580645161 | 0.6933333333 | 2 |
| tudd | 4068 | 1 | 1.33191615 | 3054.246320235 | 0.902219785 | 0.5144779491 | 0.4373177843 | 0.9525565388 | 0.321888412 | 0.6818181818 | 2 |
