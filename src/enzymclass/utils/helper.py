import os
import multiprocessing as mp
from ..model.classifier import TEClassification


def get_te(train_enz_seq_file, test_enz_seq_file, label_file,
           train_feat_dirs, test_feat_dirs,
           hyper_param_file, base_algo, k, opt, rs):
    """
    The function to get enzymclass classification object
    :param train_enz_seq_file:
    :param test_enz_seq_file:
    :param label_file:
    :param train_feat_dirs:
    :param test_feat_dirs:
    :param hyper_param_file:
    :param base_algo:
    :param k: The number of best base models for enzymclass to use in the meta model
    :param opt: whether to hyperparameter optimize each base model
    :param rs: random seed
    :return:
    """
    te = TEClassification(train_enz_seq_file, test_enz_seq_file, label_file, train_feat_dirs, test_feat_dirs,
                          hyper_param_file=hyper_param_file, random_seed=rs, n_models=k, model=base_algo, optimize=opt)
    return te


def get_n_te(train_enz_seq_file, label_file, train_feat_dirs,
             hyper_param_file, base_algo, k, opt, n):

    iter_svm = zip([train_enz_seq_file for _ in range(n)],
                   [None for _ in range(n)],
                   [label_file for _ in range(n)],
                   [train_feat_dirs for _ in range(n)],
                   [None for _ in range(n)],
                   [hyper_param_file for _ in range(n)],
                   [base_algo for _ in range(n)],
                   [k for _ in range(n)],
                   [opt for _ in range(n)],
                   range(n))

    pool = mp.Pool(mp.cpu_count())
    te_objs = list(pool.starmap(get_te, iter_svm))
    return te_objs


def store_object_validation_preds(te_obj, root_dir):
    validation_storage_dir = os.path.join(root_dir, "validation")

    for featname, featobj in zip(te_obj.feat_names, te_obj.objects):
        feat_file = os.path.join(validation_storage_dir, f"{featname}.csv")
        write_mode = "a" if os.path.exists(feat_file) else "w"
        with open(feat_file, write_mode) as f:
            f.write(",".join(list(map(str, featobj.yvalid))))
            f.write("\n")
            f.write(",".join(list(map(str, featobj.ypredvalid))))
            f.write("\n")

    ensemble_file = os.path.join(validation_storage_dir, "ensemble.csv")
    write_mode = "a" if os.path.exists(ensemble_file) else "w"
    with open(ensemble_file, write_mode) as f:
        f.write(",".join(list(map(str, te_obj.en.ytest))))
        f.write("\n")
        f.write(",".join(list(map(str, te_obj.en.preds))))
        f.write("\n")

    return
