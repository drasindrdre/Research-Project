# Research-Project
Comparison of feature selection techniques for machine learning in high dimensional, small datasets

---
title: "Honours Project"
author: "Bonolo Seleko"
date: "`r Sys.Date()`"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Loading necessary packages

```{r loading necessary packages}

library(readr)
library(dplyr)
library(tidyr)
library(data.table)
library(future)
library(future.apply)
library(DMwR)
library(DiagrammeR)
library(mlr3)
library(mlr3benchmark)
library(mlr3filters)
library(mlr3learners)
library(mlr3tuning)
library(ranger)
library(mlr3pipelines)
library(mlr3fselect)
library(smotefamily)
library(stats)
library(ggplot2)
library(paradox)
library(praznik)
library(mlr3viz)

```

##Loading processed dataset

```{r read in the processed dataset}

processed_dataset <- read.csv("Dataset/GC6_RNAseq/GC6_74_processed.csv", sep = ";")

```

## Loading counts dataset

```{r read in the counts dataset}

counts_dataset <- read.csv("Dataset/GC6_RNAseq/GC6_counts_334.csv", sep = ";")

```

## Transposing counts dataset for analysis

```{r transpose rows and columns}

counts_dataset <- transpose(counts_dataset, make.names = "Ensemble.Gene.ID", keep.names = "sample")

```

## Deleting columns in processed dataset

```{r remove unneccesary columns from the processed dataset}

processed_dataset <- processed_dataset[, -which(names(processed_dataset) %in% c("age", "code", "sex", "site", "time.from.exposure..months.", "time.to.tb..months."))]

```

## Linking the primary keys

```{r merge processed dataset with counts dataset}

counts_dataset <- merge.data.table(processed_dataset,counts_dataset, by = "sample")

```

## Exploratory data analysis

```{r check for missing values}

sum(is.na(counts_dataset))

```

```{r inspecting class distribution}

class_distribution <- table(counts_dataset$group)
class_distribution

```

```{r raw data class distribution visualization }

class_distribution_df <- as.data.frame(class_distribution)

class_plot <- ggplot(class_distribution_df, aes(x = Var1, y = Freq)) + 
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Class Distribution", x = "Groups", y = "Samples") + 
  theme_minimal()

print(class_plot)

# Set the file path for the SVG file
svg("C:/Users/bonol/OneDrive/Desktop/Honours Project/class_distribution.svg")

# Print the plot to the SVG file
print(class_plot)

# Close the SVG device
dev.off()


```

## Pre-processing

```{r extract features and target variable}

features <- counts_dataset[, !names(counts_dataset) %in% c("sample", "group")]

target_variable <- counts_dataset[, "group"]

```

```{r create a table with the target variable}

target_variable_dt <- data.table(group = target_variable)

```

```{r combine target variable and features into a sinle data table}

mlr3_data <- cbind(target_variable_dt, features)

mlr3_data <- as.data.table(mlr3_data)

```

```{r ensure target variable is a factor}

mlr3_data$group <- as.factor(mlr3_data$group)

```

## Create a classification task using mlr3

```{r create a task}

task <- mlr3::TaskClassif$new(id = "gene_expression_task", backend = mlr3_data, target = "group")

```

## Define scaling step

```{r scaling to normalize features}

scaler <- PipeOpScale$new(id = "scale", param_vals = list( scale = TRUE, center = TRUE, robust = TRUE) ) 

```

## Set number of features to select

```{r set number of features to select for model training}

n_features <- 5

```

## Define resampling strategy for cross-validation

```{r define resampling strategy using 5-fold cross-validation}

resampling <- rsmp("cv", folds = 5)

```

```{r random forest learner}

rf_learner <- lrn("classif.ranger",
mtry = to_tune(1, n_features),
num.trees = to_tune(100,200),
sample.fraction = to_tune(0.8,1),
predict_type = "prob")

```

```{r define tuner}

tuner <- tnr("random_search")

```

## Define filter methods

```{r information gain filter}

info_gain_filter <- flt("information_gain")
info_gain_filter$calculate(task)
info_gain_scores <- as.data.table(info_gain_filter)
head(info_gain_scores)

```

```{r auc filter}

auc_filter <- flt("auc")
auc_filter$calculate(task)
auc_scores <- as.data.table(auc_filter)
head(auc_scores)

```

```{r importance_rf_learner}

importance_rf_learner <- lrn("classif.ranger",
mtry = to_tune(1, n_features),
num.trees = to_tune(50,100),
sample.fraction = to_tune(0.8,1), 
importance = "impurity",
predict_type = "prob")

instance_importance_filter <- ti(
  task = task,
  learner = importance_rf_learner,
  resampling = resampling,
  measures = msr(("classif.acc")),
  terminator = trm("evals", n_evals = 10)
)

tuner_importance_rf_learner <- tuner

tuner_importance_rf_learner$optimize(instance_importance_filter)

```

```{r importance filter using random forest learner}

mini_data <- mlr3_data[, .SD, .SDcols = c("group", sample(names(features), 10))]

rf_model <- ranger(
  formula = group ~ ., 
  data = mini_data, 
  importance = "impurity"
)

importance_scores <- rf_model$variable.importance
print(importance_scores)

importance_filter <- flt("importance", learner = lrn("classif.ranger", importance = "impurity"))

importance_filter$calculate(task)

importance_scores <- as.data.table(importance_filter)
print(head(importance_scores))


```

```{r ANOVA filter}

anova_filter <- flt("anova")
anova_filter$calculate(task)
anova_scores <- as.data.table(anova_filter)
head(anova_scores)

```

## Define performance metrics

```{r define performance metrics as a list}
performance_metrics <-  list(
"Accuracy" = msr("classif.acc"),
"Recall" = msr("classif.recall"),
"Precision" = msr("classif.precision"),
"F1" = msr("classif.fbeta", beta = 1)
)


```

## Strategy 1 : Perform feature selection inside the cross-validation process

```{r infomation gain - inside cross validation process}

po_filter_info_gain_cv <- po("filter", filter = info_gain_filter, param_vals = list(filter.nfeat = n_features))

pipeline_info_gain_cv <-  scaler %>>% po_filter_info_gain_cv %>>% rf_learner

learner_info_gain_cv <- as_learner(pipeline_info_gain_cv)

info_gain_graph_cv <- learner_info_gain_cv$graph

info_gain_graph_cv$plot()


```

```{r information gain auto tuner - inside cross validation process}

#Create an instance for tuning
instance_info_gain_cv <- ti(
  learner = learner_info_gain_cv,
  task = task,
  resampling = resampling,
  measure = performance_metrics,
  terminator = trm("evals", n_evals = 10)
)

# View the tuning instance
print(instance_info_gain_cv)

# Initialize the tuner
tuner_info_gain_cv <- tnr("random_search")

# Print the tuner details 
print(tuner_info_gain_cv)

#  Optimize the tuning instance
tuner_info_gain_cv$optimize(instance_info_gain_cv)

#  Check the archive of tuning results
results_archive_info_gain_cv <- as.data.table(instance_info_gain_cv$archive)
print(results_archive_info_gain_cv)

#  Assign the best parameters to the learner
best_params_info_gain_cv <- instance_info_gain_cv$result$params

# Set a default for one of the required parameters if they are not already included
if (is.null(best_params_info_gain_cv$information_gain.filter.frac)) {
    best_params_info_gain_cv$information_gain.filter.frac <- 0.5  
}

# Set the scale.robust parameter
best_params_info_gain_cv$scale.robust <- TRUE  

# Update the learner's parameters
learner_info_gain_cv$param_set$values <- best_params_info_gain_cv

#  Train the learner with the task
learner_info_gain_cv$train(task)

# Proceed with resampling
resample_result_info_gain_cv <- resample(task, learner_info_gain_cv, resampling, store_models = TRUE)

# Aggregate performance metrics
performance_metric_info_gain_cv <- resample_result_info_gain_cv$aggregate(performance_metrics)
performance_metric_info_gain_cv


```

```{r auc - inside cross validation process}

po_filter_auc_cv <- po("filter", filter = auc_filter, param_vals = list(filter.nfeat = n_features))

pipeline_auc_cv <- scaler %>>% po_filter_auc_cv %>>% rf_learner

learner_auc_cv <- as_learner(pipeline_auc_cv, robust = TRUE)

auc_graph_cv <- learner_auc_cv$graph

auc_graph_cv$plot()

```

```{r auc auto tuner - inside cross validation process}

# Create an instance for tuning
instance_auc_cv <- ti(
  learner = learner_auc_cv,
  task = task,
  resampling = resampling,
  measure = performance_metrics,
  terminator = trm("evals", n_evals = 10)
)

# View the tuning instance
print(instance_auc_cv)

# Initialize the tuner
tuner_auc_cv <- tnr("random_search")

# Print the tuner details (optional)
print(tuner_auc_cv)

# Optimize the tuning instance
tuner_auc_cv$optimize(instance_auc_cv)

#  Check the archive of tuning results
results_archive_auc_cv <- as.data.table(instance_auc_cv$archive)
print(results_archive_auc_cv)

#  Assign the best parameters to the learner
best_params_auc_cv <- instance_auc_cv$result$params

# Set a default for one of the required parameters if they are not already included
if (is.null(best_params_auc_cv$auc.filter.frac)) {
    best_params_auc_cv$auc.filter.frac <- 0.5  
}

# Set the scale.robust parameter
best_params_auc_cv$scale.robust <- TRUE  

# Update the learner's parameters
learner_auc_cv$param_set$values <- best_params_auc_cv

#  Train the learner with the task
learner_auc_cv$train(task)

# Proceed with resampling
resample_result_auc_cv <- resample(task, learner_auc_cv, resampling, store_models = TRUE)

# Aggregate performance metrics
performance_metric_auc_cv <- resample_result_auc_cv$aggregate(performance_metrics)
performance_metric_auc_cv

```

```{r feature importance - inside cross validation process}

po_filter_importance_cv <- po("filter", filter = importance_filter, param_vals = list(filter.nfeat = n_features))

pipeline_importance_cv <-  scaler %>>% po_filter_importance_cv %>>% rf_learner

learner_importance_cv <- as_learner(pipeline_importance_cv, robust = TRUE)

importance_graph_cv <- learner_importance_cv$graph

importance_graph_cv$plot()


```

```{r feature importance auto tuner - inside corss validation process}

# Create an instance for tuning
instance_importance_cv <- ti(
  learner = learner_importance_cv,
  task = task,
  resampling = resampling,
  measure = performance_metrics,
  terminator = trm("evals", n_evals = 10)
)

# View the tuning instance
print(instance_importance_cv)

#  Initialize the tuner
tuner_importance_cv <- tnr("random_search")

# Print the tuner details 
print(tuner_importance_cv)

#  Optimize the tuning instance
tuner_importance_cv$optimize(instance_importance_cv)

# Check the archive of tuning results
results_archive_importance_cv <- as.data.table(instance_importance_cv$archive)
print(results_archive_importance_cv)

# Assign the best parameters to the learner
best_params_importance_cv <- instance_importance_cv$result$params

# Set a default for one of the required parameters if they are not already included
if (is.null(best_params_importance_cv$importance.filter.frac)) {
    best_params_importance_cv$importance.filter.frac <- 0.5  
}

# Set the scale.robust parameter
best_params_importance_cv$scale.robust <- TRUE  

# Update the learner's parameters
learner_importance_cv$param_set$values <- best_params_importance_cv

# Train the learner with the task
learner_importance_cv$train(task)

# Proceed with resampling
resample_result_importance_cv <- resample(task, learner_importance_cv, resampling, store_models = TRUE)

# Aggregate performance metrics
performance_metric_importance_cv <- resample_result_importance_cv$aggregate(performance_metrics)
performance_metric_importance_cv


```

```{r infomation gain - inside cross validation process}

po_filter_anova_cv <- po("filter", filter = anova_filter, param_vals = list(filter.nfeat = n_features))

pipeline_anova_cv <-  scaler %>>% po_filter_anova_cv %>>% rf_learner

learner_anova_cv <- as_learner(pipeline_anova_cv)

anova_graph_cv <- learner_anova_cv$graph

anova_graph_cv$plot()


```

```{r feature importance auto tuner - inside corss validation process}

# Create an instance for tuning
instance_anova_cv <- ti(
  learner = learner_anova_cv,
  task = task,
  resampling = resampling,
  measure = performance_metrics,
  terminator = trm("evals", n_evals = 10)
)

# View the tuning instance
print(instance_anova_cv)

#  Initialize the tuner
tuner_anova_cv <- tnr("random_search")

# Print the tuner details 
print(tuner_anova_cv)

#  Optimize the tuning instance
tuner_anova_cv$optimize(instance_anova_cv)

# Check the archive of tuning results
results_archive_anova_cv <- as.data.table(instance_anova_cv$archive)
print(results_archive_anova_cv)

# Assign the best parameters to the learner
best_params_anova_cv <- instance_anova_cv$result$params

# Set a default for one of the required parameters if they are not already included
if (is.null(best_params_anova_cv$anova.filter.frac)) {
    best_params_anova_cv$anova.filter.frac <- 0.5  
}

# Set the scale.robust parameter
best_params_anova_cv$scale.robust <- TRUE  

# Update the learner's parameters
learner_anova_cv$param_set$values <- best_params_anova_cv

# Train the learner with the task
learner_anova_cv$train(task)

# Proceed with resampling
resample_result_anova_cv <- resample(task, learner_anova_cv, resampling, store_models = TRUE)

# Aggregate performance metrics
performance_metric_anova_cv <- resample_result_anova_cv$aggregate(performance_metrics)
performance_metric_anova_cv


```

```{r create list of learners - inside cross validation process}

learners_list_cv <- list(
  learner_info_gain_cv,
  learner_auc_cv,
  learner_importance_cv,
  learner_anova_cv
)

```

```{r benchmark learners - inside cross validation process}

# Create benchmark grid
design_cv <- benchmark_grid(task, learners_list_cv, resampling)

head(design_cv)

# Run benchmark
bmr_cv <- benchmark(design_cv)

# Results aggregation and visualization
results_cv <- bmr_cv$aggregate(measures = performance_metrics)
print(results_cv)
autoplot(bmr_cv)
autoplot(bmr_cv, type = 'roc')

```

##  Strategy 2 : Generate new synthetic data based on thee real data to increase the sample size and then divide the combined data into traditional training/test sets

```{r generate synthetic data to increase sample size}

feature_augment <- counts_dataset[, !names(counts_dataset) %in% c("sample", "group")]

feature_augment <- as.data.frame(lapply(feature_augment, as.numeric))

target_augment <- counts_dataset$group

mlr3_data_augment <- data.frame(group = as.factor(target_augment), feature_augment)

augment_task <- mlr3::TaskClassif$new(id = "augment_gene_expression_task", backend = mlr3_data_augment, target = "group")

smote_pop_augment <- po("smote")

tsk_smote_augment <- smote_pop_augment$train(list(augment_task))[[1]]

augment_data <- tsk_smote_augment$data()

combined_data <- rbind(mlr3_data_augment, augment_data)

```

```{r check class distribution in combined dataset}

combined_class_distribution <- table(combined_data$group)
print(combined_class_distribution)

```

```{r plot class distribution for combined data}

ggplot(as.data.frame(combined_class_distribution), aes(x = Var1, y = Freq)) + 
  geom_bar(stat = "identity") + 
  labs(title = "Class Distribution of Combined Data", x = "Groups", y = "Samples") + 
  theme_minimal()

```

```{r }

set.seed(123)

train_index <- caret::createDataPartition(combined_data$group, p = 0.75, list = FALSE)

train_data <- combined_data[train_index,]

test_data <- combined_data[-train_index,]

```

```{r create classification task for train and test sets}

task_train <- mlr3::TaskClassif$new(id = "augment_gene_expression_train", backend =  train_data, target = "group")

task_test <- mlr3::TaskClassif$new(id = "augment_gene_expression_test", backend = test_data, target = "group" )

```

```{r infomation gain - for combined data}

po_filter_info_gain_augment <- po("filter", filter = info_gain_filter, param_vals = list(filter.nfeat = n_features))

pipeline_info_gain_augment <- smote_pop_augment %>>% scaler %>>% po_filter_info_gain_augment %>>% rf_learner

learner_info_gain_augment <- GraphLearner$new(pipeline_info_gain_augment)

info_gain_graph_augment <- learner_info_gain_augment$graph

info_gain_graph_augment$plot()

```

```{r information gain tuner - for combined data}

# Create an instance for tuning
instance_info_gain_augment <- ti(
  learner = learner_info_gain_augment,
  task = task_train,
  resampling = resampling,
  measure = performance_metrics,
  terminator = trm("evals", n_evals = 10)
)

# View the tuning instance
print(instance_info_gain_augment)

# Initialize the tuner
tuner_info_gain_augment <- tnr("random_search")

# Print the tuner details 
print(tuner_info_gain_augment)

# Optimize the tuning instance
tuner_info_gain_augment$optimize(instance_info_gain_augment)

# Check the archive of tuning results
results_archive_info_gain_augment <- as.data.table(instance_info_gain_augment$archive)
print(results_archive_info_gain_augment)

# Assign the best parameters to the learner
best_params_info_gain_augment <- instance_info_gain_augment$result$params

# Set a default for one of the required parameters if they are not already included
if (is.null(best_params_info_gain_augment$information_gain.filter.frac)) {
    best_params_info_gain_augment$information_gain.filter.frac <- 0.5  
}


# Set the scale.robust parameter
best_params_info_gain_augment$scale.robust <- TRUE  

# Update the learner's parameters
learner_info_gain_augment$param_set$values <- best_params_info_gain_augment

# Train the learner with the task
learner_info_gain_augment$train(task_train)

# Proceed with resampling
resample_result_info_gain_augment <- resample(task_train, learner_info_gain_augment, resampling, store_models = TRUE)

# Aggregate performance metrics
performance_metric_info_gain_augment <- resample_result_info_gain_augment$aggregate(performance_metrics)
performance_metric_info_gain_augment


```

```{r auc - for combined data}

po_filter_auc_augment <- po("filter", filter = auc_filter, param_vals = list(filter.nfeat = n_features))

pipeline_auc_augment <- smote_pop_augment %>>% scaler %>>% po_filter_auc_augment %>>% rf_learner

learner_auc_augment <- GraphLearner$new(pipeline_auc_augment)

auc_graph_augment <- learner_auc_augment$graph

auc_graph_augment$plot()


```

```{r auc auto tuner - for combined data}

# 1. Create an instance for tuning
instance_auc_augment <- ti(
  learner = learner_auc_augment,
  task = task_train,
  resampling = resampling,
  measure = performance_metrics,
  terminator = trm("evals", n_evals = 10)
)

# View the tuning instance
print(instance_auc_augment)

#  Initialize the tuner
tuner_auc_augment <- tnr("random_search")

# Print the tuner details
print(tuner_auc_augment)

# Optimize the tuning instance
tuner_auc_augment$optimize(instance_auc_augment)

# Check the archive of tuning results
results_archive_auc_augment <- as.data.table(instance_auc_augment$archive)
print(results_archive_auc_augment)

# Assign the best parameters to the learner
best_params_auc_augment <- instance_auc_augment$result$params

# Set a default for one of the required parameters if they are not already included
if (is.null(best_params_auc_augment$auc.filter.frac)) {
    best_params_auc_augment$auc.filter.frac <- 0.5  
}


# Set the scale.robust parameter
best_params_auc_augment$scale.robust <- TRUE  

# Update the learner's parameters
learner_auc_augment$param_set$values <- best_params_auc_augment

# Train the learner with the task
learner_auc_augment$train(task_train)

# Proceed with resampling
resample_result_auc_augment <- resample(task_train, learner_auc_augment, resampling, store_models = TRUE)

# Aggregate performance metrics
performance_metric_auc_augment <- resample_result_auc_augment$aggregate(performance_metrics)
performance_metric_auc_augment


```

```{r feature importance - for combined data}

po_filter_importance_augment <- po("filter", filter = importance_filter, param_vals = list(filter.nfeat = n_features))

pipeline_importance_augment <- smote_pop_augment %>>% scaler %>>% po_filter_importance_augment %>>% rf_learner

learner_importance_augment <- GraphLearner$new(pipeline_importance_augment)

importance_graph_augment <- learner_importance_augment$graph

importance_graph_augment$plot()


```

```{r feature importance auto tuner - for combined data}

#  Create an instance for tuning
instance_importance_augment <- ti(
  learner = learner_importance_augment,
  task = task_train,
  resampling = resampling,
  measure = performance_metrics,
  terminator = trm("evals", n_evals = 10)
)

# View the tuning instance
print(instance_importance_augment)

# Initialize the tuner
tuner_importance_augment <- tnr("random_search")

# Print the tuner details 
print(tuner_importance_augment)

# Optimize the tuning instance
tuner_importance_augment$optimize(instance_importance_augment)

#  Check the archive of tuning results
results_archive_importance_augment <- as.data.table(instance_importance_augment$archive)
print(results_archive_importance_augment)

#  Assign the best parameters to the learner
best_params_importance_augment <- instance_importance_augment$result$params

# Set a default for one of the required parameters if they are not already included
if (is.null(best_params_importance_augment$importance.filter.frac)) {
    best_params_importance_augment$importance.filter.frac <- 0.5  
}

# Set the scale.robust parameter
best_params_importance_augment$scale.robust <- TRUE  

# Update the learner's parameters
learner_importance_augment$param_set$values <- best_params_importance_augment

# Train the learner with the task
learner_importance_augment$train(task_train)

# Proceed with resampling
resample_result_importance_augment <- resample(task_train, learner_importance_augment, resampling, store_models = TRUE)

# Aggregate performance metrics
performance_metric_importance_augment <- resample_result_importance_augment$aggregate(performance_metrics)
performance_metric_importance_augment


```

```{r infomation gain - inside cross validation process}

po_filter_anova_augment <- po("filter", filter = anova_filter, param_vals = list(filter.nfeat = n_features))

pipeline_anova_augment <- smote_pop_augment %>>%  scaler %>>% po_filter_anova_augment %>>% rf_learner

learner_anova_augment <- as_learner(pipeline_anova_augment)

anova_graph_augment <- learner_anova_augment$graph

anova_graph_augment$plot()


```

```{r feature importance auto tuner - inside corss validation process}

# Create an instance for tuning
instance_anova_augment <- ti(
  learner = learner_anova_augment,
  task = task_train,
  resampling = resampling,
  measure = performance_metrics,
  terminator = trm("evals", n_evals = 10)
)

# View the tuning instance
print(instance_anova_augment)

#  Initialize the tuner
tuner_anova_augment <- tnr("random_search")

# Print the tuner details 
print(tuner_anova_augment)

#  Optimize the tuning instance
tuner_anova_augment$optimize(instance_anova_augment)

# Check the archive of tuning results
results_archive_anova_augment <- as.data.table(instance_anova_augment$archive)
print(results_archive_anova_augment)

# Assign the best parameters to the learner
best_params_anova_augment <- instance_anova_augment$result$params

# Set a default for one of the required parameters if they are not already included
if (is.null(best_params_anova_augment$anova.filter.frac)) {
    best_params_anova_augment$anova.filter.frac <- 0.5  
}

# Set the scale.robust parameter
best_params_anova_augment$scale.robust <- TRUE  

# Update the learner's parameters
learner_anova_augment$param_set$values <- best_params_anova_augment

# Train the learner with the task
learner_anova_augment$train(task_train)

# Proceed with resampling
resample_result_anova_augment <- resample(task_train, learner_anova_augment, resampling, store_models = TRUE)

# Aggregate performance metrics
performance_metric_anova_augment <- resample_result_anova_augment$aggregate(performance_metrics)
performance_metric_anova_augment


```

```{r create list of learners for combined data}

learners_list_augment <- list(
  learner_info_gain_augment,
  learner_auc_augment,
  learner_importance_augment,
  learner_anova_augment
)

```

```{r benchmark learners for combined data}

# Create benchmark grid
design_train <- benchmark_grid(task_train, learners_list_augment, resampling)

head(design_train)

# Run benchmark
bmr_train <- benchmark(design_train)

# Results aggregation and visualization
results_train <- bmr_train$aggregate(measures = performance_metrics)
print(results_train)
autoplot(bmr_train)
autoplot(bmr_train, type = 'roc')

```

```{r results - for combined data}

results_train <- bmr_train$aggregate(measures = performance_metrics)
results_train

```

```{r evaluate the learners on the test set}

predictions <- lapply(learners_list_augment, function(learner) {
  learner$train(task_train)
  learner$predict(task_test)
})

results_test <- sapply(predictions, function(pred) {
  pred$score(performance_metrics)
 })

results_test


```

