#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

library(kebabs)
library(Matrix)

train_fasta <- paste(args[1], "seq/train_enz.fa", sep="")
test_fasta <- paste(args[1], "seq/test_enz.fa", sep="")

train_aa <- readAAStringSet(train_fasta)
test_aa <- readAAStringSet(test_fasta)
aa <- c(train_aa, test_aa)

specK7 <- spectrumKernel(k=7, normalized=FALSE)
specFeat <- getExRep(aa, kernel=specK7, sparse=TRUE)

matCSR_spec <- as(specFeat,"dgRMatrix")
write(colnames(matCSR_spec), file = paste(args[1], "features/kernel/spectrum/colnames.txt", sep=""))
write(rownames(matCSR_spec), file = paste(args[1], "features/kernel/spectrum/rownames.txt", sep=""))
writeMM(matCSR_spec, paste(args[1], "features/kernel/spectrum/sparsematrix.txt", sep=""))


mismK3M1 <- mismatchKernel(k=3, m=1, normalized=FALSE)
mismFeat <- getExRep(aa, kernel=mismK3M1, sparse=TRUE)

matCSR_mism <- as(mismFeat,"dgRMatrix")
write(colnames(matCSR_mism), file = paste(args[1], "features/kernel/mismatch/colnames.txt", sep=""))
write(rownames(matCSR_mism), file = paste(args[1], "features/kernel/mismatch/rownames.txt", sep=""))
writeMM(matCSR_mism, file = paste(args[1], "features/kernel/mismatch/sparsematrix.txt", sep=""))

gappyK1M2 <- gappyPairKernel(k=3, m=2, normalized=FALSE)
gappyFeat <- getExRep(aa, kernel=gappyK1M2, sparse=TRUE)

matCSR_gap <- as(gappyFeat,"dgRMatrix")
write(colnames(matCSR_gap), file = paste(args[1], "features/kernel/gappy/colnames.txt", sep=""))
write(rownames(matCSR_gap), file = paste(args[1], "features/kernel/gappy/rownames.txt", sep=""))
writeMM(matCSR_gap, paste(args[1], "features/kernel/gappy/sparsematrix.txt", sep=""))
