import os
import multiprocessing as mp
from model.classifier import TEClassification


def get_te(train_enz_seq_file, test_enz_seq_file, label_file,
           train_feat_dirs, test_feat_dirs,
           hyper_param_file, base_algo, k, opt, rs, mvp):
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
    te = TEClassification(
        train_enz_seq_file, test_enz_seq_file, label_file, train_feat_dirs, test_feat_dirs,
        hyper_param_file=hyper_param_file, random_seed=rs, n_models=k, model=base_algo, 
        optimize=opt, mvp=mvp
        )
    return te


def store_object_validation_preds(te_obj, root_dir, rs):
    validation_storage_dir = os.path.join(root_dir, "validation")

    for featname, featobj in zip(te_obj.feat_names, te_obj.objects):
        feat_storage_dir = os.path.join(validation_storage_dir, featname)
        os.makedirs(feat_storage_dir, exist_ok=True)
        feat_file = os.path.join(feat_storage_dir, f"pred_{rs}.csv")
        with open(feat_file, "w") as f:
            f.write(",".join(list(map(str, featobj.yvalid))))
            f.write("\n")
            f.write(",".join(list(map(str, featobj.ypredvalid))))
            f.write("\n")

    ensemble_storage_dir = os.path.join(validation_storage_dir, "ensemble")
    os.makedirs(ensemble_storage_dir, exist_ok=True)
    ensemble_file = os.path.join(ensemble_storage_dir, f"pred_{rs}.csv")
    with open(ensemble_file, "w") as f:
        f.write(",".join(list(map(str, te_obj.en.ytest))))
        f.write("\n")
        f.write(",".join(list(map(str, te_obj.en.preds))))
        f.write("\n")

    return


def get_model_stats(root_dir, train_enz_seq_file, label_file, train_feat_dirs,
                    hyper_param_file, base_algo, k, opt, rs, store, mvp):

    # get te object
    te_obj = get_te(
        train_enz_seq_file, None, label_file, train_feat_dirs, None, 
        hyper_param_file, base_algo, k, opt, rs, mvp
        )
    # store the model predictions only if store flag is true
    if store:
        store_object_validation_preds(te_obj, root_dir, rs)
    # return ensemble model stats
    return te_obj.precision, te_obj.recall, te_obj.en.acc


def get_n_model_stats(root_dir, train_enz_seq_file, label_file, train_feat_dirs,
                      hyper_param_file, base_algo, k, opt, n, store, mvp, threads):

    iter_svm = zip([root_dir for _ in range(n)],
                   [train_enz_seq_file for _ in range(n)],
                   [label_file for _ in range(n)],
                   [train_feat_dirs for _ in range(n)],
                   [hyper_param_file for _ in range(n)],
                   [base_algo for _ in range(n)],
                   [k for _ in range(n)],
                   [opt for _ in range(n)],
                   range(n),
                   [store for _ in range(n)],
                   [mvp for _ in range(n)])

    cores = mp.cpu_count() if threads==-1 else threads
    pool = mp.Pool(cores)
    metrics = pool.starmap(get_model_stats, iter_svm)
    pool.close()
    pool.join()

    precision = [m[0] for m in metrics]
    recall = [m[1] for m in metrics]
    accuracy = [m[2] for m in metrics]
    return precision, recall, accuracy


def store_enzymclass_test_preds(test_filename, test_enznames, test_enzpreds):
    assert len(test_enznames) == len(test_enzpreds)
    with open(test_filename, "w") as f:
        for ename, epred in zip(test_enznames, test_enzpreds):
            f.write(f"{ename},{epred}\n")
    return
