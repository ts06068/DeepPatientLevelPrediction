# @file InContextEstimator.R
#
# Copyright 2026 Observational Health Data Sciences and Informatics
#
# This file is part of DeepPatientLevelPrediction
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#' Fit an In-Context Estimator
#'
#' @description Fits a pretrained binary classifier through PLP's classifier
#' workflow, registering a separate training context in each partition.
#' @details
#' `modelSettings$settings$pythonModule` names an adapter in `inst/python`.
#' The adapter provides `fit_model(features, targets, seed, ...)`,
#' `predict_proba(model, features)` returning outcome-1 probabilities,
#' `get_selected_feature_indices(model)` returning zero-based column indices,
#' and `save_model(model, path)` / `load_model(path)` using a model directory.
#' Each adapter preserves its fitted context and preprocessing when saving.
#' Inputs are non-temporal covariate matrices with binary outcomes.
#' @param trainData The training data.
#' @param modelSettings A modelSettings object.
#' @param analysisId Id of the analysis.
#' @param analysisPath Path of the analysis.
#' @param ... Extra inputs passed to PatientLevelPrediction::fitPlp().
#' @return A plpModel object.
#' @export
fitInContextEstimator <- function(trainData, modelSettings, analysisId, analysisPath, ...) {
  if ("timeId" %in% names(trainData$covariateData$covariates)) {
    stop("In-context estimators require non-temporal covariates")
  }

  # PLP's CV callbacks accept functions; persist the named entry points instead
  # of serializing closures into the model design.
  classifierSettings <- modelSettings
  classifierSettings$fitFunction <- NULL
  classifierSettings$settings$train <- trainInContextEstimator
  classifierSettings$settings$predict <- predictInContextEstimator
  classifierSettings$settings$variableImportance <- varImpInContextEstimator
  result <- PatientLevelPrediction::fitPlp(
    trainData = trainData,
    modelSettings = classifierSettings,
    analysisId = analysisId,
    analysisPath = analysisPath,
    ...
  )
  result$modelDesign$modelSettings <- modelSettings

  modelLocation <- PatientLevelPrediction::createTempModelLoc()
  dir.create(modelLocation, recursive = TRUE, showWarnings = FALSE)
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  adapter <- reticulate::import_from_path(modelSettings$settings$pythonModule, path = path)
  adapter$save_model(result$model$model, modelLocation)
  result$model <- modelLocation
  return(result)
}

trainInContextEstimator <- function(dataMatrix, labels, hyperParameters, settings) {
  if (anyNA(labels$outcomeCount) || !all(labels$outcomeCount %in% c(0, 1))) {
    stop("In-context estimators require binary outcomeCount values (0 or 1)")
  }
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  adapter <- reticulate::import_from_path(settings$pythonModule, path = path)
  trainX <- toInContextMatrix(dataMatrix)
  trainY <- as.array(labels$outcomeCount)
  parameters <- c(
    list(features = trainX, targets = trainY, seed = settings$seed),
    camelCaseToSnakeCaseNames(hyperParameters)
  )
  model <- do.call(adapter$fit_model, parameters)
  # PLP passes only the fitted object to prediction and variable-importance callbacks.
  return(list(model = model, pythonModule = settings$pythonModule))
}

#' Predict with an In-Context Estimator
#'
#' @description Returns the probability of outcome 1 in cohort row order.
#' @param plpModel The plpModel or fitted in-context estimator.
#' @param data The plpData or covariate matrix.
#' @param cohort Data frame identifying the prediction rows.
#' @return The cohort with predicted probabilities in the value column.
#' @export
predictInContextEstimator <- function(plpModel, data, cohort) {
  if (inherits(data, "plpData")) {
    matrixObjects <- PatientLevelPrediction::toSparseM(
      plpData = data,
      cohort = cohort,
      map = plpModel$covariateImportance %>%
        dplyr::select("columnId", "covariateId")
    )
    data <- matrixObjects$dataMatrix
    cohort <- matrixObjects$labels
  }

  if (inherits(plpModel, "plpModel")) {
    pythonModule <- plpModel$modelDesign$modelSettings$settings$pythonModule
  } else {
    pythonModule <- plpModel$pythonModule
  }
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  adapter <- reticulate::import_from_path(pythonModule, path = path)
  model <- plpModel$model
  if (is.character(model)) {
    model <- adapter$load_model(model)
  }

  prediction <- cohort
  predictionX <- toInContextMatrix(data)
  prediction$value <- as.numeric(adapter$predict_proba(model, predictionX))
  prediction <- prediction %>%
    dplyr::select(-"rowId") %>%
    dplyr::rename(rowId = "originalRowId")
  attr(prediction, "metaData")$modelType <- "binary"
  return(prediction)
}

toInContextMatrix <- function(data) {
  denseSize <- prod(as.double(dim(data))) * 8 / 1024^2
  ParallelLogger::logInfo(paste0(
    "In-context estimator dense input: ", nrow(data), " x ", ncol(data),
    " (", round(denseSize, 2), " MiB per float64 matrix)"
  ))
  return(reticulate::r_to_py(as.matrix(data)))
}

varImpInContextEstimator <- function(model, covariateMap) {
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  adapter <- reticulate::import_from_path(model$pythonModule, path = path)
  selectedColumns <- adapter$get_selected_feature_indices(model$model)
  covariateMap$included <- as.integer(
    covariateMap$columnId %in% (selectedColumns + 1L)
  )
  # Selected columns are distinct from importance scores, which PLP defaults to zero.
  covariateMap$covariateValue <- 0
  return(covariateMap %>% dplyr::select("covariateId", "covariateValue", "included"))
}
