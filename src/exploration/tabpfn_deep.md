default_params:
  prediction_batch_size=2048
  fit_mode = "fit_with_cache"

---

## BASELINE
MODEL_PARAMS = {
    "n_estimators": 8,
}


| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabpfn-3 | 26.185548827 | false | 0.9955764597 | 0.9652593563 | 0.8152932961 | 0.977468748 | 0.6932897862 | 0.9894067797 | 2 |



| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 8.480637034 | 1310.6326748143 | 0.905160082 | 0.5276577609 | 0.378280543 | 0.9381916329 | 0.2593052109 | 0.6989966555 | 2 |
| tudd | 4068 | 1 | 3.093198528 | 1315.143519944 | 0.8984919787 | 0.49790325 | 0.3848580442 | 0.9520648968 | 0.2618025751 | 0.7261904762 | 2 |

---

## Testing


MODEL_PARAMS = {
    "n_estimators": 8,
    "inference_config": {
        "POLYNOMIAL_FEATURES":5,
    }
}


Tested but same performance: 
MODEL_PARAMS = {
    "n_estimators": 8,
    "inference_config": {
        "POLYNOMIAL_FEATURES":5,
        "ENABLE_GPU_PREPROCESSING":True
    }
}


| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabpfn-3 | 35.311025563 | false | 0.995316671 | 0.9582458298 | 0.7824686941 | 0.9741039675 | 0.6493467933 | 0.9842484248 | 2 |

| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 8.559486255 | 1298.5592439624 | 0.9049151083 | 0.5258080562 | 0.3848920863 | 0.9384615385 | 0.2655086849 | 0.6993464052 | 2 |
| tudd | 4068 | 1 | 3.114176459 | 1306.2843591419 | 0.8981081187 | 0.4876949367 | 0.3824451411 | 0.9515732547 | 0.2618025751 | 0.7093023256 | 2 |


---

MODEL_PARAMS = {
    "n_estimators": 8,
    "inference_config": {
        "PREPROCESS_TRANSFORMS": [
                {
                    "name": "power",
                    "categorical_name": "none",   
                }
            ]

    }
}


| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabpfn-3 | 39.154351915 | false | 0.9933555723 | 0.9497406529 | 0.7574592715 | 0.9717826948 | 0.6143111639 | 0.9875894988 | 2 |


| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 8.464510392 | 1313.1297009813 | 0.9053576978 | 0.5272579289 | 0.3754578755 | 0.9386414755 | 0.2543424318 | 0.7167832168 | 2 |
| tudd | 4068 | 1 | 3.079056385 | 1321.1839899452 | 0.9008309505 | 0.5006420848 | 0.394984326 | 0.9525565388 | 0.2703862661 | 0.7325581395 | 2 |

---

MODEL_PARAMS = {
    "n_estimators": 8,
    "inference_config": {
        "PREPROCESS_TRANSFORMS": [
                {
                    "name": "power",
                    "categorical_name": "none",   
                },
                {
                    "name": "quantile_uni_coarse",
                    "categorical_name": "none",   
                }
            ]

    }
}

| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabpfn-3 | 36.466789224 | false | 0.9945349923 | 0.9587488312 | 0.7913848345 | 0.975040995 | 0.6600356295 | 0.988 | 2 |


| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 8.454088998 | 1314.7484019425 | 0.9047750803 | 0.5272779533 | 0.3773928897 | 0.938551507 | 0.2568238213 | 0.7113402062 | 2 |
| tudd | 4068 | 1 | 3.067355673 | 1326.2237685079 | 0.9015785262 | 0.5031220247 | 0.3785488959 | 0.9515732547 | 0.2575107296 | 0.7142857143 | 2 |



---

Basically same but with GPU preprocessing -> faster fit, same predict speed
MODEL_PARAMS = {
    "n_estimators": 8,
    "inference_config": {
        "ENABLE_GPU_PREPROCESSING": True,
        "PREPROCESS_TRANSFORMS": [
                {
                    "name": "power",
                    "categorical_name": "none",   
                },
                {
                    "name": "quantile_uni_coarse",
                    "categorical_name": "none",   
                }
            ]

    }
}


| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabpfn-3 | 28.186757148 | false | 0.9945349923 | 0.9587488312 | 0.7913848345 | 0.975040995 | 0.6600356295 | 0.988 | 2 |


| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 8.445448394 | 1316.0935312677 | 0.9047750803 | 0.5272779533 | 0.3773928897 | 0.938551507 | 0.2568238213 | 0.7113402062 | 2 |
| tudd | 4068 | 1 | 3.07641768 | 1322.3171958886 | 0.9015785262 | 0.5031220247 | 0.3785488959 | 0.9515732547 | 0.2575107296 | 0.7142857143 | 2 |


---

## SUBSAMPLE_SAMPLES
MODEL_PARAMS = {
    "n_estimators": 8,
    "inference_config": {
        "SUBSAMPLE_SAMPLES": 0.5,
    }
}
-> each estimator only has 50% of samples.
-> roughly 40-50% faster


| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabpfn-3 | 8.593233482 | false | 0.9858122274 | 0.8692934725 | 0.5799302278 | 0.9564069255 | 0.4195368171 | 0.9388704319 | 2 |

| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 4.904408701 | 2266.3282523196 | 0.9041723643 | 0.5200804144 | 0.3479069767 | 0.9369320738 | 0.2320099256 | 0.6951672862 | 2 |
| tudd | 4068 | 1 | 1.768833499 | 2299.8207588788 | 0.8941945375 | 0.4783822059 | 0.3486842105 | 0.9513274336 | 0.2274678112 | 0.7464788732 | 2 |



MODEL_PARAMS = {
    "n_estimators": 8,
    "inference_config": {
        "SUBSAMPLE_SAMPLES": 0.3,
    }
}
-> 3x over baseline, slightly degraded performance

| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabpfn-3 | 4.046238477 | false | 0.9764932781 | 0.7814783809 | 0.4729876329 | 0.9482718232 | 0.3236342043 | 0.8783239323 | 2 |


| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 3.444111084 | 3227.2478235782 | 0.9033948991 | 0.5126178564 | 0.34082397 | 0.9366621682 | 0.2258064516 | 0.6946564885 | 2 |
| tudd | 4068 | 1 | 1.248209116 | 3259.06929204 | 0.8838146505 | 0.4487956382 | 0.3118644068 | 0.9500983284 | 0.1974248927 | 0.7419354839 | 2 |



MODEL_PARAMS = {
    "n_estimators": 16,
    "inference_config": {
        "SUBSAMPLE_SAMPLES": 0.3,
    }
}
-> 50% faster predicts than baseline, much faster fitting too.
-> somewhat small performace hit on tudd (generalazability?)

| model | fit_time_s | tuned | train_roc_auc | train_prc_auc | train_f1 | train_accuracy | train_sensitivity | train_precision | train_n_classes |
|---|---|---|---|---|---|---|---|---|---|
| tabpfn-3 | 7.949813059 | false | 0.977504301 | 0.7805893664 | 0.4670161643 | 0.9480375663 | 0.3173990499 | 0.8834710744 | 2 |


| dataset | rows | predict_repeats | predict_time_mean_s | rows_per_second_mean | roc_auc | prc_auc | f1 | accuracy | sensitivity | precision | n_classes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic | 11115 | 1 | 6.915212581 | 1607.3258587219 | 0.9035913114 | 0.5143916236 | 0.3364661654 | 0.9364822312 | 0.2220843672 | 0.6937984496 | 2 |
| tudd | 4068 | 1 | 2.479284645 | 1640.7958675515 | 0.885927559 | 0.4523354446 | 0.3208191126 | 0.9510816126 | 0.2017167382 | 0.7833333333 | 2 |
