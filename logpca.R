#! /usr/bin/Rscript
print("Import packages...")
library(logisticPCA)
library(rARPACK)

# Find the appropriate values for k and m.
# > logsvd_model = logisticSVD(binary_df, k = 20)
# 8418 rows and 368 columns
# Rank 20 solution
# 95.6% of deviance explained
# 397 iterations to converge
#
# > logpca_cv = cv.lpca(binary_df, ks = 20, ms = 1:10)
#      m
#  k       1        2      3        4        5        6        7        8        9       10
#  20 400428 261586.6 185985 143663.3 118547.4 102668.9 92638.51 85579.33 80440.14 76707.54

print("Import data...")
binary_train <- read.csv(file="../data/mercedes/binary_train.csv", header=TRUE, sep=",")
binary_test <- read.csv(file="../data/mercedes/binary_test.csv", header=TRUE, sep=",")

print("Define parameters...")
k <- 2; m <- 4
print("Remove constant columns...")
binary_train <- binary_train[,apply(binary_train, 2, var, na.rm=TRUE) != 0]
print("Fit model...")
logpca_model = convexLogisticPCA(binary_train, k = k, m = m, quiet=FALSE, partial_decomp=TRUE)
print("Predict on train...")
logpca_features_train <- predict(logpca_model, newdata=binary_train); colnames(logpca_features_train) <- paste0("LPC", 1:k)
print("Predict on test...")
logpca_features_test <- predict(logpca_model, newdata=binary_test); colnames(logpca_features_test) <- paste0("LPC", 1:k)

print("Write data...")
write.csv(logpca_features_train, file="../data/mercedes/logpca_train.csv", row.names=FALSE)
write.csv(logpca_features_test, file="../data/mercedes/logpca_test.csv", row.names=FALSE)
