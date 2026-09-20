# Use Python >= 3.11 with exaonetabular installed and set RETICULATE_PYTHON
# before starting R. Install the runtime with:
# uv pip install --python /path/to/python \
#   "git+https://github.com/LGAI-Research/EXAONE-Tabular.git@8638e07d09fad154249bd75ba4786181491f7025"
library(DeepPatientLevelPrediction)

connectionDetails <- Eunomia::getEunomiaConnectionDetails()
Eunomia::createCohorts(connectionDetails)

covariateSettings <- FeatureExtraction::createCovariateSettings(
  useDemographicsGender = TRUE,
  useDemographicsAge = TRUE,
  useConditionOccurrenceLongTerm = TRUE
)
databaseDetails <- PatientLevelPrediction::createDatabaseDetails(
  connectionDetails = connectionDetails,
  cdmDatabaseSchema = "main",
  cdmDatabaseId = "1",
  cohortDatabaseSchema = "main",
  cohortTable = "cohort",
  targetId = 4,
  outcomeIds = 3,
  outcomeDatabaseSchema = "main",
  outcomeTable = "cohort",
  cdmDatabaseName = "eunomia"
)
populationSettings <- PatientLevelPrediction::createStudyPopulationSettings(
  requireTimeAtRisk = FALSE,
  riskWindowStart = 1,
  riskWindowEnd = 365
)
plpData <- PatientLevelPrediction::getPlpData(
  databaseDetails = databaseDetails,
  covariateSettings = covariateSettings,
  restrictPlpDataSettings = PatientLevelPrediction::createRestrictPlpDataSettings()
)

modelSettings <- setExaoneTabular(device = "cuda", seed = 0L)
plpResults <- PatientLevelPrediction::runPlp(
  plpData = plpData,
  outcomeId = 3,
  modelSettings = modelSettings,
  analysisId = "FirstModel",
  analysisName = "Testing EXAONE-Tabular",
  populationSettings = populationSettings,
  saveDirectory = "results/EXAONETabular"
)

population <- PatientLevelPrediction::createStudyPopulation(
  plpData = plpData,
  outcomeId = 3,
  populationSettings = populationSettings
)
prediction <- plpResults$prediction
evaluationStatistics <- plpResults$performanceEvaluation$evaluationStatistics
populationRows <- match(prediction$rowId, population$rowId)
stopifnot(
  is.data.frame(prediction),
  nrow(prediction) > 0,
  !anyNA(populationRows),
  all(prediction$outcomeCount == population$outcomeCount[populationRows]),
  all(is.finite(prediction$value)),
  all(prediction$value >= 0 & prediction$value <= 1),
  !anyDuplicated(prediction[c("rowId", "evaluationType")]),
  all(c("Train", "CV", "Test") %in% prediction$evaluationType),
  is.data.frame(evaluationStatistics),
  nrow(evaluationStatistics) > 0
)

savedResults <- PatientLevelPrediction::loadPlpResult(
  "results/EXAONETabular/FirstModel/plpResult"
)
testPrediction <- prediction[prediction$evaluationType == "Test", ]
# The official ensemble assigns jitter by query position, so keep query order.
predictionPopulation <- population[
  match(testPrediction$rowId, population$rowId),
]
predictionPopulation$outcomeCount <- NULL
restoredPrediction <- PatientLevelPrediction::predictPlp(
  plpModel = savedResults$model,
  plpData = plpData,
  population = predictionPopulation
)
expectedValues <- testPrediction$value[
  match(restoredPrediction$rowId, testPrediction$rowId)
]
stopifnot(
  identical(restoredPrediction$rowId, predictionPopulation$rowId),
  isTRUE(all.equal(restoredPrediction$value, expectedValues, tolerance = 1e-6))
)
print(table(prediction$evaluationType))
print(evaluationStatistics)
cat("Saved model reproduced held-out probabilities in the original query order.\n")
