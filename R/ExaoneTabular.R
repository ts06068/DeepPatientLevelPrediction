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
    fitFunction = "DeepPatientLevelPrediction::fitInContextEstimator",
    param = list(list(device = device)),
    settings = list(
      modelName = "EXAONETabular",
      pythonModule = "ExaoneTabular",
      modelType = "binary",
      seed = as.integer(seed),
      prepareData = "toSparseM",
      requiresDenseMatrix = TRUE,
      predict = "DeepPatientLevelPrediction::predictInContextEstimator",
      saveType = "file"
    )
  )
  class(results) <- "modelSettings"
  return(results)
}

#' Fit EXAONE-Tabular
#'
#' @description Fits EXAONE-Tabular through the shared in-context estimator.
#' Retained for model settings created before the shared estimator was introduced.
#' @param trainData The training data.
#' @param modelSettings A modelSettings object.
#' @param analysisId Id of the analysis.
#' @param analysisPath Path of the analysis.
#' @param ... Extra inputs passed to PatientLevelPrediction::fitPlp().
#' @return A plpModel object.
#' @export
fitExaoneTabular <- function(trainData, modelSettings, analysisId, analysisPath, ...) {
  modelSettings$settings$pythonModule <- "ExaoneTabular"
  fitInContextEstimator(trainData, modelSettings, analysisId, analysisPath, ...)
}

#' Predict with EXAONE-Tabular
#'
#' @description Returns the probability of outcome 1 in cohort row order.
#' Retained so previously saved EXAONE-Tabular models can still be used.
#' @param plpModel The plpModel or fitted Python classifier.
#' @param data The plpData or covariate matrix.
#' @param cohort Data frame identifying the prediction rows.
#' @return The cohort with predicted probabilities in the value column.
#' @export
predictExaoneTabular <- function(plpModel, data, cohort) {
  if (inherits(plpModel, "plpModel")) {
    plpModel$modelDesign$modelSettings$settings$pythonModule <- "ExaoneTabular"
  } else {
    plpModel <- list(model = plpModel, pythonModule = "ExaoneTabular")
  }
  predictInContextEstimator(plpModel, data, cohort)
}
