import os
import argparse
import numpy as np
import logging
import utils.preprocess as utp
import utils.features as utf
import utils.helper as uth


def main(
    root_dir, train_file, test_file, base_algo, k_best_models, n_sim, store, mvp, threads, featurize_only, 
    predict_only):
    # make the appropriate directory structure for proper EnZymClass functioning
    utp.make_dirs(root_dir)
    # log file configurations
    log_filename =  os.path.join(root_dir, "logs/enzymclass.log")
    logging.basicConfig(filename=log_filename, format="%(asctime)s;%(levelname)s;%(message)s", encoding='utf-8', filemode="w", level=logging.INFO)
    # preprocess the raw data files provided in csv format
    logging.info('Preprocessing raw data files ...')
    enz_train_name_map, enz_test_name_map, enz_train_fasta, enz_test_fasta, enz_train_csv, enz_test_csv, enz_train_labels = utp.raw_data_parsing(
        root_dir, train_file, test_file)
    # define feature file output paths
    ifeat_enz_train_output = os.path.join(root_dir, "features/ifeatpro/train/")
    ifeat_enz_test_output = os.path.join(root_dir, "features/ifeatpro/test/")
    pssm_enz_train_output = os.path.join(root_dir, "features/pssmpro/train/")
    pssm_enz_test_output = os.path.join(root_dir, "features/pssmpro/test/")
    kernel_enz_train_output = os.path.join(root_dir, "features/kernel/train/")
    kernel_enz_test_output = os.path.join(root_dir, "features/kernel/test/")

    if not predict_only:
        # create ifeatpro features
        logging.info('Creating ifeatpro features ...')
        utf.create_ifeatpro_features(enz_train_fasta, ifeat_enz_train_output, enz_test_fasta, ifeat_enz_test_output)
        # pssmpro preprocessing
        uniref_path = os.path.join(root_dir, "features/pssmpro/")
        pssm_profile_train_path = os.path.join(root_dir, "features/pssmpro/pssm_profiles/train/")
        pssm_profile_test_path = os.path.join(root_dir, "features/pssmpro/pssm_profiles/test/")
        logging.info('Creating pssm profiles ...')
        utf.create_pssm_profiles(uniref_path, enz_train_csv, enz_test_csv, pssm_profile_train_path, pssm_profile_test_path, nthreads=threads)
        # create pssmpro features
        logging.info('Creating pssmpro features ...')
        utf.create_pssmpro_features(pssm_profile_train_path, pssm_enz_train_output, pssm_profile_test_path, pssm_enz_test_output)
        # create kernel features
        logging.info('Creating kernel features ...')
        utf.create_kernel_features(root_dir)

    train_feature_dirs = [ifeat_enz_train_output,
                          pssm_enz_train_output,
                          kernel_enz_train_output]
    test_feature_dirs = [ifeat_enz_test_output,
                         pssm_enz_test_output,
                         kernel_enz_test_output]

    if not featurize_only:
        # run enzymclass validation N times to give the user an idea of it's performance
        # if store, store model results in validation directory
        if n_sim>0:
            logging.info(f'Training {n_sim} models to generate average validation performance ...')
            precision, recall, accuracy = uth.get_n_model_stats(root_dir, enz_train_csv, enz_train_labels, train_feature_dirs,
                                                            None, base_algo, k_best_models, False, n_sim, store, mvp, threads)

            # output ensemble model statistics
            logging.info("EnZymClass statistics on validation for multiple runs:")
            logging.info("|----------------|-------------|--------------|")
            logging.info("| Mean Precision | Mean Recall | Mean Accuracy |")
            logging.info(f"|******{round(np.mean(precision), 2)}******|*****{round(np.mean(recall), 2)}*****|*****{round(np.mean(accuracy), 2)}*****|")
            logging.info("|----------------|-------------|--------------|")

        # predict using enzymclass
        logging.info('Training EnZymClass and predicting test data labels ...')
        te_pred = uth.get_te(
            enz_train_csv, enz_test_csv, enz_train_labels, train_feature_dirs, test_feature_dirs, 
            hyper_param_file=None, base_algo=base_algo, k=k_best_models, opt=False, rs=None, mvp=mvp
            )
        logging.info(f"EnZymClass statistics on validation for the current run:")
        logging.info(f"Accuracy: {te_pred.en_valid.acc}")
        logging.info(f"Precision: {te_pred.precision}")
        logging.info(f"Recall: {te_pred.recall}")
        # store the predictions in predictions/ directory
        test_pred_file = os.path.join(root_dir, "predictions/enzymclass_preds.csv")
        enz_test_name_dict = {line.strip().split(",")[0]: line.strip().split(",")[1] for line in open(enz_test_name_map, "r").readlines()}
        test_enz_original_names = [enz_test_name_dict[te_alias] for te_alias in te_pred.test_enz_names]
        logging.info(f'Storing test labels in "{test_pred_file}" ...')
        uth.store_enzymclass_test_preds(test_pred_file, test_enz_original_names, te_pred.en.preds)

    logging.info("Thank you for using EnZymClass ...")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnZymClass: Ensemble model for enzyme classification")
    parser.add_argument("root_dir", type=str, help="Provide the root directory where EnZymClass output will be stored")
    parser.add_argument("train_file", type=str, help="Train file path. CSV file with the convention: enz_name,enz_sequence,enz_label")
    parser.add_argument("test_file", type=str, help="Test file path. CSV file with the convention: enz_name,enz_sequence")

    parser.add_argument("-b", "--base", type=str, default="SVM", help="base algorithm: either SVM/NN/GBC")
    parser.add_argument("-n", "--nsim", type=int, default=1000, help="number of simulations for validation")
    parser.add_argument("-k", "--nmod", type=int, default=5, help="number of base models for meta learner prediction")
    parser.add_argument("-c", "--iclass", type=int, default=None, help="most important label of interest to calculate validation scores")
    parser.add_argument("-t", "--threads", type=int, default=-1, help="Number of parallel processors to use")

    parser.add_argument("--featurize", "-f", help="Will only create features and not run the predictive model",
                    action="store_true")
    parser.add_argument("--predict", "-p", help="Will only run the predictive model, assumes that the features have been created and stored in appropriate dirs and subdirs",
                    action="store_true")
    parser.add_argument("--store", "-s", help="Will store validation predictions of each base model if called",
                        action="store_true")

    args = parser.parse_args()

    main(args.root_dir, args.train_file, args.test_file, args.base, args.nmod, args.nsim, args.store, args.iclass, args.threads, args.featurize, args.predict)
