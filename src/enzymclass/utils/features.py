import os
import gzip
import shutil
import subprocess
import numpy as np
from scipy import sparse, io
import ifeatpro.features as ipro
import pssmpro.features as ppro

FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)


def create_ifeatpro_features(enz_train_fasta, enz_train_output, enz_test_fasta, enz_test_output):
    ipro.get_all_features(enz_train_fasta, enz_train_output)
    ipro.get_all_features(enz_test_fasta, enz_test_output)
    return


def make_uniref_db(uniref_prefix):
    # download and unzip uniref50 database from uniprot
    uniref_url = "http://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
    uniref_path = os.path.join(uniref_prefix, "uniref50.fasta")
    subprocess.call(["wget", "-P", f"{uniref_prefix}", f"{uniref_url}"])
    with gzip.open(f'{os.path.join(uniref_prefix, "uniref50.fasta.gz")}', 'rb') as f_in:
        with open(f'{uniref_path}', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    # create a blast database
    subprocess.call(["makeblastdb", "-in", uniref_path, "-dbtype", "prot", "-out", f"{uniref_prefix}uniref50"])
    return


def create_pssm_profiles_helper(seq_file, out_dir, database_prefix, nthreads=8):
    for line in open(seq_file).readlines():
        parsed_line = line.strip().split(",")
        name, sequence = parsed_line[0], parsed_line[1]
        with open(name + '.txt', 'w') as f:
            f.write('>' + name + '\n' + sequence + '\n')
        if not os.path.exists(os.path.join(out_dir, name + '.pssm')):
            print("Generating psiblast profile for protein: " + name)
            subprocess.call([f"psiblast", "-query", f"{name}.txt", "-db", f"{database_prefix}", "-num_iterations", "3",
                             "-num_threads", f"{nthreads}", "-out_ascii_pssm", f"{os.path.join(out_dir, name + '.pssm')}"])
        os.remove(name + '.txt')
    return


def create_pssm_profiles(uniref_prefix, train_seq_file, test_seq_file, train_out_dir, test_out_dir, nthreads=8):
    # uniref database creation
    uniref_path = os.path.join(uniref_prefix, "uniref50.fasta")
    if not os.path.exists(uniref_path):
        make_uniref_db(uniref_prefix)

    # create the profiles
    uniref_db_prefix = uniref_prefix + "uniref50"
    create_pssm_profiles_helper(train_seq_file, train_out_dir, uniref_db_prefix, nthreads)
    create_pssm_profiles_helper(test_seq_file, test_out_dir, uniref_db_prefix, nthreads)
    return


def create_pssmpro_features(enz_train_profiles, enz_train_output, enz_test_profiles, enz_test_output):
    ppro.get_all_features(enz_train_profiles, enz_train_output)
    ppro.get_all_features(enz_test_profiles, enz_test_output)
    return


def parse_kernel_matrix(specific_kernel_dir, enz_train_output, enz_test_output):
    """
    converts the raw files created by kebabs for a specific kernel to numpy compatible matrix
    :param specific_kernel_dir: The directory path for a specific kernel output from kebab
    :param enz_train_output: The prefix path to store the kernel specific features of the train enzyme 
    :param enz_test_output: The prefix path to store the kernel specific features of the train enzyme
    :return: 
    """
    sp_mat_file = os.path.join(specific_kernel_dir, 'sparsematrix.txt')
    enz_name_file = os.path.join(specific_kernel_dir, 'rownames.txt')

    sp_mat = io.mmread(sp_mat_file).tocsr()
    enz_names = np.genfromtxt(enz_name_file, dtype=str)

    train_enz_idx = []
    test_enz_idx = []

    for idx, enz_name in enumerate(enz_names):
        if enz_name.startswith('enz'):
            train_enz_idx.append(idx)
        elif enz_name.startswith('test'):
            test_enz_idx.append(idx)
        else:
            raise ValueError('Wrong Enzyme Prefix')

    X_train, X_test = sp_mat[train_enz_idx, :], sp_mat[test_enz_idx, :]

    enz_names_train, enz_names_test = enz_names[train_enz_idx], enz_names[test_enz_idx]

    assert X_train.shape[0] == len(enz_names_train)
    assert X_test.shape[0] == len(enz_names_test)

    sparse.save_npz(enz_train_output + 'mat.npz', X_train)
    sparse.save_npz(enz_test_output + 'mat.npz', X_test)

    np.savetxt(enz_train_output + 'enz_names.txt', enz_names_train, fmt='%s')
    np.savetxt(enz_test_output + 'enz_names.txt', enz_names_test, fmt='%s')
    return


def create_kernel_features(root_dir):
    # call the rscript that calls kebabs to create kernel features
    subprocess.call(["Rscript", f"{FILE_DIR}/kernels.R", f"{root_dir}"])
    # parse the kebabs output for downstream analysis
    kernel_types = ["spectrum", "mismatch", "gappy"]
    kernel_prefixes = ["spec", "mism", "gap"]
    for i, kernel in enumerate(kernel_types):
        specific_dir = os.path.join(root_dir, "features/kernel/", kernel) # root_dir + "features/kernel/" + kernel
        enz_train_out_prefix = os.path.join( root_dir, "features/kernel/train/", kernel_prefixes[i]) # root_dir + "features/kernel/train/" + kernel_prefixes[i]
        enz_test_out_prefix = os.path.join(root_dir, "features/kernel/test/", kernel_prefixes[i]) # root_dir + "features/kernel/test/" + kernel_prefixes[i]
        parse_kernel_matrix(specific_dir, enz_train_out_prefix, enz_test_out_prefix)
    return
