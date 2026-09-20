# @file ExaoneTabular.R
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

#' Create EXAONE-Tabular Settings
#'
#' @description
#' Creates one fixed configuration for the official pretrained EXAONE-Tabular
#' classifier. Fitting registers training examples as in-context support without
#' gradient training.
#'
#' @details
#' Requires Python >= 3.11 and the official `exaonetabular` package installed in
#' the Python environment used by reticulate. Only non-temporal covariates are
#' supported. Sparse zeros remain zero and explicit missing values remain missing.
#' The official runtime performs preprocessing and feature selection within each
#' training partition. Saved models include the fitted context and preprocessing.
#'
#' @param device Device used by EXAONE-Tabular, for example "cuda" (default) or "cpu".
#' @param seed Random seed for the model.
#' @return A modelSettings object for PatientLevelPrediction::runPlp().
#' @export
setExaoneTabular <- function(device = "cuda", seed = 0L) {
  checkIsClass(device, "character")
  checkIsClass(seed, c("integer", "numeric"))
  if (length(device) != 1 || length(seed) != 1 || !is.finite(seed) ||
      seed < 0 || seed > .Machine$integer.max || seed != floor(seed)) {
    stop("device and seed must be single values, with a non-negative integer seed")
  }

  results <- list(
    fitFunction = "DeepPatientLevelPrediction::fitExaoneTabular",
    param = list(list(device = device)),
    settings = list(
      modelName = "EXAONETabular",
      modelType = "binary",
      seed = as.integer(seed),
      prepareData = "toSparseM",
      requiresDenseMatrix = TRUE,
      predict = "DeepPatientLevelPrediction::predictExaoneTabular",
      saveType = "file"
    )
  )
  class(results) <- "modelSettings"
  return(results)
}

#' Fit EXAONE-Tabular
#'
#' @description Fits EXAONE-Tabular through PLP's classifier workflow.
#' @param trainData The training data.
#' @param modelSettings A modelSettings object.
#' @param analysisId Id of the analysis.
#' @param analysisPath Path of the analysis.
#' @param ... Extra inputs passed to PatientLevelPrediction::fitPlp().
#' @return A plpModel object.
#' @export
fitExaoneTabular <- function(trainData, modelSettings, analysisId, analysisPath, ...) {
  if ("timeId" %in% names(trainData$covariateData$covariates)) {
    stop("EXAONE-Tabular requires non-temporal covariates")
  }

  # PLP's CV callbacks accept functions; persist the named entry points instead
  # of serializing closures into the model design.
  classifierSettings <- modelSettings
  classifierSettings$fitFunction <- NULL
  classifierSettings$settings$train <- trainExaoneTabular
  classifierSettings$settings$predict <- predictExaoneTabular
  classifierSettings$settings$variableImportance <- varImpExaoneTabular
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
  exaone <- reticulate::import_from_path("ExaoneTabular", path = path)
  exaone$save_exaone_tabular(
    result$model,
    file.path(modelLocation, "ExaoneTabularModel.pt")
  )
  result$model <- modelLocation
  return(result)
}

trainExaoneTabular <- function(dataMatrix, labels, hyperParameters, settings) {
  if (anyNA(labels$outcomeCount) || !all(labels$outcomeCount %in% c(0, 1))) {
    stop("EXAONE-Tabular requires binary outcomeCount values (0 or 1)")
  }
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  exaone <- reticulate::import_from_path("ExaoneTabular", path = path)
  trainX <- toExaoneMatrix(dataMatrix)
  trainY <- as.array(labels$outcomeCount)
  model <- exaone$fit_exaone_tabular(
    features = trainX,
    targets = trainY,
    device = hyperParameters$device,
    seed = settings$seed
  )
  return(model)
}

#' Predict with EXAONE-Tabular
#'
#' @description Returns the probability of outcome 1 in cohort row order.
#' @param plpModel The plpModel or fitted Python classifier.
#' @param data The plpData or covariate matrix.
#' @param cohort Data frame identifying the prediction rows.
#' @return The cohort with predicted probabilities in the value column.
#' @export
predictExaoneTabular <- function(plpModel, data, cohort) {
  path <- system.file("python", package = "DeepPatientLevelPrediction")
  exaone <- reticulate::import_from_path("ExaoneTabular", path = path)
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
    model <- plpModel$model
    if (is.character(model)) {
      model <- exaone$load_exaone_tabular(
        file.path(model, "ExaoneTabularModel.pt")
      )
    }
  } else {
    model <- plpModel
  }

  prediction <- cohort
  predictionX <- toExaoneMatrix(data)
  prediction$value <- as.numeric(exaone$predict_exaone_tabular(model, predictionX))
  prediction <- prediction %>%
    dplyr::select(-"rowId") %>%
    dplyr::rename(rowId = "originalRowId")
  attr(prediction, "metaData")$modelType <- "binary"
  return(prediction)
}

toExaoneMatrix <- function(data) {
  denseSize <- prod(as.double(dim(data))) * 8 / 1024^2
  ParallelLogger::logInfo(paste0(
    "EXAONE-Tabular dense input: ", nrow(data), " x ", ncol(data),
    " (", round(denseSize, 2), " MiB per float64 matrix)"
  ))
  return(reticulate::r_to_py(as.matrix(data)))
}

varImpExaoneTabular <- function(model, covariateMap) {
  selectedColumns <- model$selected_feature_indices_
  if (is.null(selectedColumns)) {
    selectedColumns <- seq_len(nrow(covariateMap)) - 1L
  }
  covariateMap$included <- as.integer(
    covariateMap$columnId %in% (selectedColumns + 1L)
  )
  # The public API exposes selected columns, not variable importance scores.
  # PLP uses zero when a model does not provide importance scores.
  covariateMap$covariateValue <- 0
  return(covariateMap %>% dplyr::select("covariateId", "covariateValue", "included"))
}
