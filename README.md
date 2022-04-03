# EnZymClass
Please refer to the following link for now: 
https://github.com/deeprob/ThioesteraseEnzymeSpecificity

Deepro Banerjee, Michael A. Jindra, Alec J. Linot, Brian F. Pfleger, Costas D. Maranas,
**EnZymClass: Substrate specificity prediction tool of plant acyl-ACP thioesterases based on ensemble learning**,
Current Research in Biotechnology,
Volume 4,
2022,
Pages 1-9,
ISSN 2590-2628,
https://doi.org/10.1016/j.crbiot.2021.12.002.

# Usage
## Step 1: Environment creation
```bash
foo@bar:~$ conda create -n enzymclass -c conda-forge -c bioconda -c anaconda python=3.9 scikit-learn pandas multiprocess blast wget bioconductor-kebabs
foo@bar:~$ conda activate enzymclass
foo@bar:~$ pip install ngrampro ifeatpro pssmpro
```

## Step 2: EnZymClass input and output
### Required input arguments
1. **root_dir**: *The directory path where all intermediate and final output files generated by EnZymClass will be stored. Look at the section **"Output directory structure"** for more details.*

2. **train_file**: *The path to the csv file which contains information about the training sequences required to train EnZymClass. For more details see **"Train file format"** section.*

3. **test_file**: *The path to the csv file which contains information about the test sequences whose labels will be predicted by EnZymClass. For more details see **"Test file format"** section.*

### Train file format
The train file should be a csv file without any headers. The format is as follows:

```
protein_unique_name,protein_sequence,protein_numerical_category
```

For example:

```
A._hypogaea_l._(AhFatA),MLKVSCNGSDRVQFMAQCGFAGQPASVLVRRRSVSAVGFGYPMNRVLSVRAIVSDRDGAVVNRVGAEAGTLADRLRLGSLTEDGLSYKEKFIVRSYEVGINKTATVETIANLLQEVGCNHAQSVGYSTDGFATTPTMRKLGLIWVTARMHIEVYKYPAWSDVVEIETWCQGEGRVGIRRDFILKDYATDQVIGRATSKWLMMNQETRRLQKVSDDVREEVLIYCPREPRLAIPEEDSNCLKKIPKLEDPGQYSRLRLMPRRADLDMNQHVNNVTYIGWVLESMPQEIIDSHELHSITLDYRRECQRDDIVDSLTSIEGDGVLLEVNGTNGSSVAWEHGHAYQQFLHLLKLSTDEGLEINRGRTAWRKKASRL,1
.
.
.
```

### Test file format
The test file should be a csv file without any headers. The format is as follows:

```
protein_unique_name,protein_sequence
```

For example:

```
Uncharacterized_protein__ECO_0000313_EMBL_EMT12172.1_,MAGSVASGFFPTPGSSPAASARGSKNMSGELPESLSVRGMVAKPNTPPASMQVKARAQALPKVNGSKVNLKTTGSDKEDTVPYTSSKTFYNQLPDWSMLLAAVTTIFLAAEKQWTMLDWKPKRPDMLVDTFGFGRIIQDGLVFRQNFLIRSYEIGADRTASIETLMNHLQETALNHVKTAGLLGDGFGATPEMSKRNLIWVVSKIQLLVEHYPSWEDMVQVDTWVASAGKNGMRRDWHIRDYNSGRTILKATSVWVMMNKTTRRLSKMPDEVRGEIGPHFNDRSAITEEQGEKLAKPRNKVVDPANKQFIRKGLTPKWGDLDVNQHVNNVKYIGWILESAPISILEKHELASMTLDYRKECCRDSVLQSLTNVSGECVDGSPDSAIQCDHLLQLESGADVVKAHTTWRPKRAHGEGNLGLFPVESA
.
.
.
```

### Output directory structure
EnZymClass will create the following directory structure to store all intermediate and final output files.

```
📦root
 ┣ 📂features
 ┣ 📂label
 ┣ 📂mappings
 ┣ 📂predictions
 ┣ 📂seq
 ┗ 📂validation
```

Here, *root* refers to the root directory path provided by the user as the first required argument of EnZymClass. Under root there are 6 directories created by EnZymClass whose contents are defined as follows:

1. *features*: The protein sequences provided to EnZymClass are numerically encoded and stored in this directory. To know the type of feature encodings used by EnZymClass please go through our paper.
2. *label*: The training protein alias and their corresponding numerical categories are stored here.
3. *mappings*: Protein aliases created by EnZymClass mapped to their original user provided names are stored here. 
4. *predictions*: EnZymClass' test set predictions are stored here. For the format of this file please see section *Output file format*.
5. *seq*:  Protein aliases mapped to their user provided sequences are stored here.
6. *validation*: EnZymClass validation dataset predictions for N runs are stored here. For each run, EnZymClass creates a different training and validation set and assesses its own performance based on these multiple runs. We refer the reader to our paper for more details.

### Output file format
The output prediction file is a csv file without any headers. The format is as follows:

```
protein_unique_name,predicted_numerical_category
```

For example:

```
Uncharacterized_protein__ECO_0000313_EMBL_EMT12172.1_,1
.
.
.
```

## Step 3: Running EnZymClass
```bash
# clone this repo
foo@bar:~$ git clone https://github.com/deeprob/EnZymClass.git
# change directory to repo's root
foo@bar:~$ cd /path/to/EnZymClass
# activate the conda environment created in Step 1
foo@bar:~$ conda activate enzymclass
# run EnZymClass
foo@bar:~$ python src/enzymclass/run_model.py /path/to/root_dir /path/to/train.csv /path/to/test.csv
```

# Additional tips
## Downloading uniref database beforehand for faster result generation
*EnZymClass checks if the uniref database has already been downloaded and is present in the "/path/to/root_dir/pssmpro" directory. If not present, then it will download and store the database on the said directory. Since it is a huge file, an user can download it prior to running EnZymClass and store in the directory mentioned above as uniref50.fasta. After downloading the uniref database, an user will also have to convert it to a blast compatible database using the "makeblastdb" command as mentioned below:*
```bash
foo@bar:~$ makeblastdb -in "/path/to/root_dir/pssmpro/uniref50.fasta" -dbtype "prot" -out "/path/to/root_dir/pssmpro/uniref50"
```

## Passing user defined number of cores to EnZymClass
*Users can pass the number of cores to be used by EnZymClass using the "--threads" optional argument. By default EnZymClass uses maximum available threads.*

## Reducing the number of validation simulations to run 
*EnZymClass estimates an accurate validation performance by simulating N number of models where each model uses unique training and validation sets. By default, N=1000. To get faster results, an user can reduce the number of simulations to run by specifying a lower number through the "--nsim" optional argument. To stop EnZymClass from generating this report, "--nsim" can be set to 0.*

## Using EnZymClass to **ONLY CREATE FEATURES**.
*To only create features and not run the prediction model, run enzymclass with the "--featurize" argument.*

# Future Work
1. Packaging EnZymClass.
2. Incorporating feature embedders in EnZymClass.
