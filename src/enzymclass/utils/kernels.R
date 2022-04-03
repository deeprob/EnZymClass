#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

library(kebabs)
library(Matrix)

train_fasta <- file.path(args[1], "seq/train_enz.fa")
test_fasta <- file.path(args[1], "seq/test_enz.fa")

train_aa <- readAAStringSet(train_fasta)
test_aa <- readAAStringSet(test_fasta)
aa <- c(train_aa, test_aa)

specK7 <- spectrumKernel(k=7, normalized=FALSE)
specFeat <- getExRep(aa, kernel=specK7, sparse=TRUE)

matCSR_spec <- as(specFeat,"dgRMatrix")
write(colnames(matCSR_spec), file = file.path(args[1], "features/kernel/spectrum/colnames.txt"))
write(rownames(matCSR_spec), file = file.path(args[1], "features/kernel/spectrum/rownames.txt"))
writeMM(matCSR_spec, file.path(args[1], "features/kernel/spectrum/sparsematrix.txt"))


mismK3M1 <- mismatchKernel(k=3, m=1, normalized=FALSE)
mismFeat <- getExRep(aa, kernel=mismK3M1, sparse=TRUE)

matCSR_mism <- as(mismFeat,"dgRMatrix")
write(colnames(matCSR_mism), file = file.path(args[1], "features/kernel/mismatch/colnames.txt"))
write(rownames(matCSR_mism), file = file.path(args[1], "features/kernel/mismatch/rownames.txt"))
writeMM(matCSR_mism, file = file.path(args[1], "features/kernel/mismatch/sparsematrix.txt"))

gappyK1M2 <- gappyPairKernel(k=3, m=2, normalized=FALSE)
gappyFeat <- getExRep(aa, kernel=gappyK1M2, sparse=TRUE)

matCSR_gap <- as(gappyFeat,"dgRMatrix")
write(colnames(matCSR_gap), file = file.path(args[1], "features/kernel/gappy/colnames.txt"))
write(rownames(matCSR_gap), file = file.path(args[1], "features/kernel/gappy/rownames.txt"))
writeMM(matCSR_gap, file.path(args[1], "features/kernel/gappy/sparsematrix.txt"))
