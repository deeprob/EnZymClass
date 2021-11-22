import os
import pandas as pd


def make_dirs(root_dir):
    """
    Make the appropriate directory structure required for proper EnZymClass functioning
    :param root_dir: The root directory where all other files and directories will be stored
    :return:
    """
    assert os.path.isdir(root_dir)
    print(f"EnZymClass will create and store files here: {os.path.abspath(root_dir)}")

    # create mappings, seq, label, features, predictions directories
    os.makedirs(os.path.join(root_dir, "mappings"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "seq"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "label"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "features"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "validation"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "predictions"), exist_ok=True)

    # create ifeatpro, kernel, pssmpro dir inside features
    os.makedirs(os.path.join(root_dir, "features/ifeatpro/train"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "features/ifeatpro/test"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "features/kernel/train"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "features/kernel/test"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "features/kernel/spectrum"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "features/kernel/mismatch"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "features/kernel/gappy"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "features/pssmpro/train"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "features/pssmpro/test"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "features/pssmpro/pssm_profiles/train"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "features/pssmpro/pssm_profiles/test"), exist_ok=True)
    return


def up_seq(seq):
    return seq.upper().replace('-', '')


def raw_data_parsing(root_dir, train_raw, test_raw):
    """
    Given train and test raw filepath, this function processes the data for downstream analysis
    :param root_dir:
    :param train_raw:
    :param test_raw:
    :return:
    """
    df_train = pd.read_csv(train_raw, header=None, names=["enz_name", "enz_seq", "enz_label"])
    df_test = pd.read_csv(test_raw, header=None, names=["enz_name", "enz_seq"])

    # upper case all sequences
    df_train["enz_seq"] = df_train.enz_seq.apply(up_seq)
    df_test["enz_seq"] = df_test.enz_seq.apply(up_seq)

    # get rid of sequences with illegitimate amino acids
    df_train = df_train.loc[~df_train["enz_seq"].str.contains('B|J|O|U|X|Z')]
    df_test = df_test.loc[~df_test["enz_seq"].str.contains('B|J|O|U|X|Z')]

    # create enzyme alias
    enz_alias_train = [f'enz_{i}' for i in range(len(df_train['enz_name']))]
    df_train = df_train.assign(enz_alias=enz_alias_train)
    enz_alias_test = [f'test_enz_{i}' for i in range(len(df_test['enz_name']))]
    df_test = df_test.assign(enz_alias=enz_alias_test)

    # enzyme alias to original enzyme name mapping file creation
    enz_train_name_map = os.path.join(root_dir, "mappings/train_enz_map.csv")
    enz_test_name_map = os.path.join(root_dir, "mappings/test_enz_map.csv")
    df_train.loc[:, ["enz_alias", "enz_name"]].to_csv(enz_train_name_map, index=False, header=False)
    df_test.loc[:, ["enz_alias", "enz_name"]].to_csv(enz_test_name_map, index=False, header=False)

    # fasta file creation
    enz_train_fasta = os.path.join(root_dir, "seq/train_enz.fa")
    enz_test_fasta = os.path.join(root_dir, "seq/test_enz.fa")
    train_fasta_stream = open(enz_train_fasta, "w")
    test_fasta_stream = open(enz_test_fasta, "w")
    for value in df_train.loc[:, ["enz_alias", "enz_seq"]].values:
        train_fasta_stream.write(f">{value[0]}\n{value[1]}\n")
    for value in df_test.loc[:, ["enz_alias", "enz_seq"]].values:
        test_fasta_stream.write(f">{value[0]}\n{value[1]}\n")
    train_fasta_stream.close()
    test_fasta_stream.close()

    # sequence csv file creation
    enz_train_csv = os.path.join(root_dir, "seq/train_enz.csv")
    enz_test_csv = os.path.join(root_dir, "seq/test_enz.csv")
    df_train.loc[:, ["enz_alias", "enz_seq"]].to_csv(enz_train_csv, header=False, index=False)
    df_test.loc[:, ["enz_alias", "enz_seq"]].to_csv(enz_test_csv, header=False, index=False)

    # label csv file creation
    enz_train_labels = os.path.join(root_dir, "label/train_enz_label.csv")
    df_train.loc[:, ["enz_alias", "enz_label"]].to_csv(enz_train_labels, index=False, header=False)

    return enz_train_name_map, enz_test_name_map, enz_train_fasta, enz_test_fasta, enz_train_csv, enz_test_csv, enz_train_labels
