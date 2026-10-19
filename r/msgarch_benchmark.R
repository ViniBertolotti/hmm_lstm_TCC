# Benchmark econométrico MS-GARCH (Haas, Mittnik e Paolella, 2004)
# Implementação: pacote MSGARCH (Ardia et al., 2019)
#
# Execute da raiz, depois de scripts/msgarch_inputs.py:
#   Rscript r/msgarch_benchmark.R
#
# Para cada janela walk-forward:
#   1. ajusta MS-GARCH(1,1) Student-t com K = 2 e K = 3 estados (FitML) só nos
#      cinco anos de desenvolvimento (treino + validação + calibração);
#   2. escolhe K pelo BIC, ainda só no desenvolvimento;
#   3. com parâmetros congelados, filtra a série até o fim do teste. A previsão para
#      o pregão t usa só retornos até t-1 (verificado: alterar y_t não muda a linha t);
#   4. exporta a vol condicional prevista e os resíduos padronizados do
#      desenvolvimento (para o VaR/ES por FHS, calculado em Python com a mesma
#      função dos outros modelos) e o VaR/ES paramétrico da densidade preditiva
#      (análise de sensibilidade).
#
# Retornos em log, em %, sem média (como a referência de média zero).

suppressMessages({library(MSGARCH); library(arrow)})

dir <- "reports/tables/msgarch"
returns <- as.data.frame(read_parquet(file.path(dir, "inputs_returns.parquet")))
folds <- as.data.frame(read_parquet(file.path(dir, "inputs_folds.parquet")))
targets <- as.data.frame(read_parquet(file.path(dir, "inputs_targets.parquet")))

states_grid <- c(2L, 3L)
levels <- c(0.95, 0.99)
grid <- seq(-30, 30, by = 0.01)  # log-retorno em %
burn_in <- 21L                   # resíduos iniciais dependem da variância inicial

make_spec <- function(K) CreateSpec(variance.spec = list(model = "sGARCH"),
                                    distribution.spec = list(distribution = "std"),
                                    switch.spec = list(K = K))

state_unconditional_var <- function(par, K) {
  sapply(seq_len(K), function(k) par[[paste0("alpha0_", k)]] /
           (1 - par[[paste0("alpha1_", k)]] - par[[paste0("beta_", k)]]))
}

fits_out <- list(); fc_out <- list(); res_out <- list()
for (i in seq_len(nrow(folds))) {
  f <- folds[i, ]
  t0 <- Sys.time()
  window <- returns[returns$date >= f$dev_first & returns$date <= f$test_last & !is.na(returns$return_1d), ]
  dev <- window$date <= f$dev_last
  y_all <- 100 * window$return_1d
  y_dev <- y_all[dev]

  fits <- list()
  for (K in states_grid) {
    fit <- tryCatch(FitML(make_spec(K), data = y_dev), error = function(e) NULL)
    ok <- !is.null(fit) && all(is.finite(fit$par))
    fits[[as.character(K)]] <- fit
    fits_out[[length(fits_out) + 1]] <- data.frame(
      fold = f$fold, K = K, converged = ok,
      loglik = if (ok) as.numeric(fit$loglik) else NA_real_,
      bic = if (ok) as.numeric(BIC(fit)) else NA_real_,
      n_dev = length(y_dev),
      par = if (ok) as.character(jsonlite::toJSON(as.list(fit$par), digits = 10, auto_unbox = TRUE)) else NA_character_)
  }
  table_f <- do.call(rbind, Filter(function(x) x$fold[1] == f$fold, fits_out))
  K <- table_f$K[which.min(table_f$bic)]
  fit <- fits[[as.character(K)]]
  spec <- make_spec(K)
  par <- fit$par

  vol <- as.numeric(Volatility(spec, par = par, data = y_all))
  dens <- PredPdf(spec, x = grid, par = par, data = y_all, do.its = TRUE)
  pred_prob <- State(spec, par = par, data = y_all)$PredProb[seq_along(y_all), 1, , drop = TRUE]
  order_by_var <- order(state_unconditional_var(par, K))  # 1 = estado de menor variância
  high_state <- order_by_var[K]

  # Resíduos padronizados do desenvolvimento (FHS)
  z <- y_dev / vol[dev]
  keep <- seq_along(z) > burn_in
  res_out[[i]] <- data.frame(fold = f$fold, date = window$date[dev][keep], z = z[keep])

  # Previsões para os alvos do teste
  tg <- targets[targets$fold == f$fold, ]
  rows <- match(tg$target, window$date)
  if (anyNA(rows)) stop(sprintf("janela %d: alvo sem retorno observado", f$fold))
  step <- diff(grid[1:2])
  loss <- -expm1(grid / 100)  # perda = -retorno simples
  par_risk <- t(sapply(rows, function(r) {
    p <- dens[r, ] * step
    p <- p / sum(p)
    # perdas ordenadas da maior para a menor (grid crescente em retorno)
    cum <- cumsum(p)
    out <- c()
    for (a in levels) {
      tail <- 1 - a
      j <- which(cum >= tail)[1]
      w <- p[1:j]; w[j] <- w[j] - (cum[j] - tail)
      out <- c(out, loss[j], sum(w * loss[1:j]) / tail)
    }
    out
  }))
  colnames(par_risk) <- c("var95_param", "es95_param", "var99_param", "es99_param")
  fc_out[[i]] <- data.frame(fold = f$fold, origin = tg$origin, target = tg$target, K = K,
                            vol_pct = vol[rows], prob_high_state = pred_prob[rows, high_state],
                            par_risk)
  cat(sprintf("janela %d: K=%d (BIC %s) | %d alvos | %.0fs\n", f$fold, K,
              paste(round(table_f$bic, 1), collapse = "/"), nrow(tg),
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  flush.console()
}

write_parquet(do.call(rbind, fits_out), file.path(dir, "msgarch_fits.parquet"))
write_parquet(do.call(rbind, fc_out), file.path(dir, "msgarch_forecasts.parquet"))
write_parquet(do.call(rbind, res_out), file.path(dir, "msgarch_residuals.parquet"))
writeLines(c(R.version.string, paste("MSGARCH", packageVersion("MSGARCH"))), file.path(dir, "versions.txt"))
