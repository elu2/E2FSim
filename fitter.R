library(data.table)
library(glmnet)

n <- 10000000
test_n <- 10000

# Read in full dataset
data <- fread("C:/Users/UAL-Laptop/Downloads/SM111111121.csv", showProgress = TRUE, fill=TRUE)[, c(1:27)]


# neit - dataframe with neither bistable or rebi
# secondary - dataframe with either bistable or rebi
# n - number of total rows
# prop - proportion of neit to secondary data rows

enrich_combine <- function(neit, secondary, n_rows, prop){
  n1 <- ceiling(n_rows * prop)
  n2 <- ceiling(n_rows * (1 - prop))
  
  rNeitIndices <- sample(nrow(neit), n1)
  rBiSubsetIndices <- sample(nrow(biSubset), n2)
  
  df_neit <- neit[rNeitIndices, ]
  df_bi <- biSubset[rBiSubsetIndices, ]
  
  combined <- rbind(df_neit, df_bi)
  
  return(combined)
}


# Function that takes data and partitions out x, y, and the data unsampled from

train_set <- function(data, n) {
  last_col <- ncol(data)
  train_indices <- sample(nrow(data), n)
  
  tr_x <- as.matrix(data[train_indices, 1:(last_col-1)])
  tr_y <- as.matrix(rev(data)[train_indices, 1])
  
  remainder <- data[-train_indices,]
  
  return(list(tr_x, tr_y, remainder))
}


# data: dataframe of data to sample from

fitter <- function(data, n){
  train_data <- train_set(data, n)
  tr_x <- train_data[[1]]
  tr_y <- train_data[[2]]
  
  fit <- glmnet(tr_x, tr_y, family="binomial")
  cv_fit <- cv.glmnet(tr_x, tr_y, family="binomial", type.measure="class")
  
  return(list(fit, cv_fit))
}


# fit: fit object
# data: data to sample (randomly) from for newx
# s: specific lambda value

prediction_fun <- function(fit, data, s){
  test_indices <- sample(nrow(data), test_n)
  last_col <- ncol(data)
  te_x <- as.matrix(data[test_indices, 1:(last_col)])
  
  return(predict(fit, newx=te_x, s=s, type="class"))
}


fit <- fitter(data, n)
save(fit, file="./fit.RData")
