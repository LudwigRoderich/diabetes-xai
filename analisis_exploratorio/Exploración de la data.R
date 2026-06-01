library(isotree)
library(tidyverse)
library(rpart)
library(rpart.plot)
library(parallel)
library(mRMRe)

# Carga y preparación inicial
ruta <- "C:/Users/rluis/Códigos/Proyectos/Diabetes + XAI/data/"
archivo <- "diabetes_binary_health_indicators_BRFSS2015.csv"
file <- paste0(ruta, archivo)

diabetes_data <- read.csv(file, header = TRUE, sep = ",")
caracteristicas <- diabetes_data[, -1]
objetivo <- diabetes_data[, 1]

binary_vars <- names(caracteristicas)[sapply(caracteristicas, function(col) {
  vals <- unique(na.omit(col))
  length(vals) == 2 && all(vals %in% c(0, 1))
})]

categorical_vars <- unique(c(
  binary_vars,
  "GenHlth",
  "Age",
  "Education",
  "Income"
))

for (col in categorical_vars) {
  caracteristicas[[col]] <- as.factor(caracteristicas[[col]])
}

# Detección de outliers con Isolation Forest
N <- 1000
n_observaciones <- nrow(diabetes_data)
umbral_anomalia <- 0.6
umbral_estabilidad <- ceiling(0.75 * N)

arbol_i <- function(i) {
  model_iforest_iteration <- isolation.forest(
    data = caracteristicas,
    ntrees = 100,
    ndim = 1,
    categ_split_type = "single_categ",
    sample_size = 256,
    nthreads = 1,
    seed = i
  )
  
  anomaly_scores <- predict(
    model_iforest_iteration,
    caracteristicas,
    type = "score"
  )
  
  indices_anomalias <- which(anomaly_scores > umbral_anomalia)
  cantidad_anomalias <- length(indices_anomalias)
  porcentaje_anomalias <- (cantidad_anomalias / n_observaciones) * 100
  
  list(
    resumen = data.frame(
      cantidad_anomalias = cantidad_anomalias,
      porcentaje_anomalias = porcentaje_anomalias
    ),
    indices_anomalias = indices_anomalias
  )
}

num_cores <- 8
cl <- makeCluster(num_cores)

clusterExport(cl, c(
  "caracteristicas",
  "n_observaciones",
  "umbral_anomalia",
  "arbol_i"
))

clusterEvalQ(cl, {
  library(isotree)
})

resultados <- parLapply(cl, 1:N, arbol_i)
stopCluster(cl)

resultados_df <- do.call(
  rbind,
  lapply(resultados, function(x) x$resumen)
)

todos_los_indices <- unlist(
  lapply(resultados, function(x) x$indices_anomalias)
)

frecuencia_indices <- table(todos_los_indices)

outliers_estables <- as.integer(
  names(
    frecuencia_indices[
      frecuencia_indices >= umbral_estabilidad
    ]
  )
)

cat(
  "Cantidad de outliers estables:",
  length(outliers_estables),
  "\n"
)
cat(
  "Porcentaje eliminado:",
  round(length(outliers_estables) / nrow(diabetes_data) * 100, 4),
  "%\n"
)
diabetes_data_sin_outliers <- diabetes_data[-outliers_estables, ]
write.csv(
  diabetes_data_sin_outliers,
  "C:/Users/rluis/Downloads/diabetes/diabetes_data_sin_outliers.csv",
  row.names = FALSE
)

cantidades_anomalias <- resultados_df$cantidad_anomalias
porcentajes_anomalias <- resultados_df$porcentaje_anomalias
media_cantidad <- mean(cantidades_anomalias)
sd_cantidad <- sd(cantidades_anomalias)
media_porcentaje <- mean(porcentajes_anomalias)
sd_porcentaje <- sd(porcentajes_anomalias)

cat("\n========== RESUMEN ==========" , "\n")
cat("Media de anomalías:", round(media_cantidad, 2), "\n")
cat("Desviación estándar de anomalías:", round(sd_cantidad, 2), "\n")
cat("Media del porcentaje:", round(media_porcentaje, 4), "%\n")
cat("Desviación estándar del porcentaje:", round(sd_porcentaje, 4), "%\n")

resultados_df <- data.frame(
  iteracion = 1:N,
  cantidad_anomalias = cantidades_anomalias,
  porcentaje_anomalias = porcentajes_anomalias
)

# Visualización del comportamiento de anomalías
ggplot(resultados_df, aes(x = cantidad_anomalias)) +
  geom_histogram(bins = 15) +
  geom_vline(
    xintercept = media_cantidad,
    linetype = "dashed"
  ) +
  labs(
    title = "Distribución de la cantidad de anomalías",
    x = "Cantidad de anomalías",
    y = "Frecuencia"
  )

ggplot(resultados_df, aes(y = cantidad_anomalias)) +
  geom_boxplot() +
  labs(
    title = "Variabilidad de la cantidad de anomalías",
    y = "Cantidad de anomalías"
  )

ggplot(resultados_df, aes(x = porcentaje_anomalias)) +
  geom_histogram(bins = 15) +
  geom_vline(
    xintercept = media_porcentaje,
    linetype = "dashed"
  ) +
  labs(
    title = "Distribución del porcentaje de anomalías",
    x = "Porcentaje de anomalías",
    y = "Frecuencia"
  )

ggplot(resultados_df, aes(y = porcentaje_anomalias)) +
  geom_boxplot() +
  labs(
    title = "Variabilidad del porcentaje de anomalías",
    y = "Porcentaje de anomalías"
  )

# Selección de variables con mRMR
mrmr_data <- diabetes_data_sin_outliers
mrmr_data = diabetes_data[,-23] # Quitar el anomaly_score

for (col in names(mrmr_data)) {
  if (is.factor(mrmr_data[[col]])) {
    mrmr_data[[col]] <- as.numeric(as.character(mrmr_data[[col]]))
  }
}

mrmr_data <- mutate_all(mrmr_data, as.numeric)

mrmr_dataset <- mRMR.data(data = mrmr_data)

mrmr_resultados <- mRMR.classic(
  data = mrmr_dataset,
  target_indices = 1,
  feature_count = 23
)

variables_seleccionadas <- solutions(mrmr_resultados)[[1]]
scores_mrmr = round(scores(mrmr_resultados)[[1]], 6)
causality_mrmr = round(causality(mrmr_resultados)[[1]], 6)
ranking_mrmr <- data.frame(
  indice = variables_seleccionadas,
  variable = colnames(mrmr_data)[variables_seleccionadas],
  Score = scores_mrmr[variables_seleccionadas],
  Causality = causality_mrmr[variables_seleccionadas]
)
ranking_mrmr = ranking_mrmr[order(ranking_mrmr$Score, decreasing = TRUE),]
ranking_mrmr = ranking_mrmr[ranking_mrmr$indice != 1,]

print(ranking_mrmr)
