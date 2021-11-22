import os
import argparse
import numpy as np
from .utils.preprocess import make_dirs, raw_data_parsing
from .utils.features import create_ifeatpro_features, create_pssm_profiles, create_pssmpro_features, create_kernel_features
from .utils.helper import get_te, get_n_model_stats


def main(root_dir, train_file, test_file, base_algo, k_best_models, n_sim, store, debug=True):
    # make the appropriate directory structure for proper EnZymClass functioning
    make_dirs(root_dir)
    # preprocess the raw data files provided in csv format
    enz_train_name_map, enz_test_name_map, enz_train_fasta, enz_test_fasta, enz_train_csv, enz_test_csv, enz_train_labels = raw_data_parsing(
        root_dir, train_file, test_file)
    # define feature file output paths
    ifeat_enz_train_output = os.path.join(root_dir, "features/ifeatpro/train/")
    ifeat_enz_test_output = os.path.join(root_dir, "features/ifeatpro/test/")
    pssm_enz_train_output = os.path.join(root_dir, "features/pssmpro/train/")
    pssm_enz_test_output = os.path.join(root_dir, "features/pssmpro/test/")
    kernel_enz_train_output = os.path.join(root_dir, "features/kernel/train/")
    kernel_enz_test_output = os.path.join(root_dir, "features/kernel/test/")

    if not debug:
        # create ifeatpro features
        create_ifeatpro_features(enz_train_fasta, ifeat_enz_train_output, enz_test_fasta, ifeat_enz_test_output)
        # create pssmpro features
        uniref_path = os.path.join(root_dir, "features/pssmpro/")
        pssm_profile_train_path = os.path.join(root_dir, "features/pssmpro/pssm_profiles/train/")
        pssm_profile_test_path = os.path.join(root_dir, "features/pssmpro/pssm_profiles/test/")
        create_pssm_profiles(uniref_path, enz_train_csv, enz_test_csv, pssm_profile_train_path, pssm_profile_test_path)
        create_pssmpro_features(pssm_profile_train_path, pssm_enz_train_output, pssm_profile_test_path, pssm_enz_test_output)
        # create kernel features
        create_kernel_features(root_dir)

    train_feature_dirs = [ifeat_enz_train_output,
                          pssm_enz_train_output,
                          kernel_enz_train_output]
    test_feature_dirs = [ifeat_enz_test_output,
                         pssm_enz_test_output,
                         kernel_enz_test_output]

    # run enzymclass validation N times to give the user an idea of it's performance
    # TODO: if store, store model results in validation directory
    precision, recall, accuracy = get_n_model_stats(root_dir, enz_train_csv, enz_train_labels, train_feature_dirs,
                                                    None, base_algo, k_best_models, False, n_sim, store)

    # TODO: create ensemble model statistics
    print("|----------------|-------------|--------------|")
    print("| Mean Precision | Mean Recall | Mean Accuracy |")
    print(f"|******{round(np.mean(precision), 2)}******|*****{round(np.mean(recall), 2)}*****|*****{round(np.mean(accuracy), 2)}*****|")
    print("|----------------|-------------|--------------|")

    # predict using enzymclass
    # te_pred = get_te(enz_train_csv, enz_test_csv, enz_train_labels, train_feature_dirs, test_feature_dirs,
    #                  None, "SVM", 5, False, 0)
    # TODO:  store the predictions in predictions/ directory
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnZymClass: Ensemble model for enzyme classification")
    parser.add_argument("root_dir", type=str, help="Provide the root directory where EnZymClass output will be stored")
    parser.add_argument("train_file", type=str, help="Train file path. CSV file with the convention: enz_name,enz_sequence,enz_label")
    parser.add_argument("test_file", type=str, help="Test file path. CSV file with the convention: enz_name,enz_sequence")

    parser.add_argument("--base", "-b", type=str, default="SVM", help="base algorithm: either SVM/NN/GBC")
    parser.add_argument("--nsim", "-n", type=int, default=1000, help="number of simulations for validation")
    parser.add_argument("--nmod", "-k", type=int, default=5, help="number of base models for meta learner prediction")

    parser.add_argument("--store", "-s", help="Will store validation predictions of each base model if called",
                        action="store_true")

    args = parser.parse_args()

    main(args.root_dir, args.train_file, args.test_file, args.base, args.nmod, args.nsim, args.store)
