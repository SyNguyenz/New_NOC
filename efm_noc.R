suppressMessages(library(euroformix))
IO <- "C:/Tailieu/TinSinh/Project/new_NOC/forensim_io"
args <- commandArgs(trailingOnly = TRUE)
per <- ifelse(length(args) >= 1, as.integer(args[1]), 1L)   # samples per NOC
AT <- 150

# popFreq
freq <- read.csv(file.path(IO, "freq.csv"), colClasses = c("character", "character", "numeric"))
popFreq <- list()
for (loc in unique(freq$locus)) {
  s <- freq[freq$locus == loc, ]; v <- s$freq; names(v) <- s$allele
  popFreq[[loc]] <- v / sum(v)
}
loci <- names(popFreq)

mh <- read.csv(file.path(IO, "mixh_test.csv"),
               colClasses = c("integer", "character", "character", "numeric"))
trueN <- as.integer(read.csv(file.path(IO, "noc_test.csv"))$noc)

build_sample <- function(sid) {
  prof <- list()
  sub <- mh[mh$sample == sid, ]
  for (loc in loci) {
    r <- sub[sub$locus == loc, ]
    if (nrow(r) > 0) prof[[loc]] <- list(adata = r$allele, hdata = r$height)
  }
  list(prof)                                  # samples = list of 1 evidence profile
}

get_loglik <- function(fit) {
  for (nm in c("loglik")) if (!is.null(fit[[nm]])) return(as.numeric(fit[[nm]]))
  if (!is.null(fit$fit$loglik)) return(as.numeric(fit$fit$loglik))
  NA_real_
}

est_noc <- function(sid, dbg = FALSE) {
  samples <- build_sample(sid); names(samples) <- "E1"
  aic <- rep(NA_real_, 5); best_aic <- Inf
  for (nC in 1:5) {
    fit <- tryCatch(
      calcMLE(nC = nC, samples = samples, popFreq = popFreq, DEG = FALSE, BWS = FALSE,
              FWS = FALSE, AT = AT, pC = 0.05, lambda = 0.01, fst = 0, nDone = 1,
              normalize = TRUE, verbose = FALSE, maxThreads = 0),
      error = function(e) { if (dbg) cat("   nC", nC, "ERR:", conditionMessage(e), "\n"); NULL })
    if (dbg && nC == 1 && !is.null(fit)) { cat("=== calcMLE result names ===\n"); print(names(fit)) }
    if (!is.null(fit)) aic[nC] <- -2 * get_loglik(fit) + 2 * (nC + 1)
    # early stop: once AIC has risen above the running best (past the minimum), stop
    if (!is.na(aic[nC])) {
      if (aic[nC] < best_aic) best_aic <- aic[nC]
      else if (nC >= 2) break                  # AIC increased -> minimum already passed
    }
  }
  if (all(is.na(aic))) return(NA_integer_)
  which.min(aic)
}

allsid <- sort(unique(mh$sample))
pick <- integer(0)
for (k in 1:5) pick <- c(pick, head(allsid[trueN[allsid + 1] == k], per))
cat("running euroformix contLikSearch on", length(pick), "samples ...\n")

t0 <- Sys.time(); out <- data.frame(sample = pick, true = trueN[pick + 1], efm = NA_integer_)
for (i in seq_along(pick)) {
  out$efm[i] <- est_noc(pick[i], dbg = (i == 1))
  cat(sprintf("  sid %d true=%d efm=%s\n", pick[i], out$true[i], out$efm[i]))
}
cat(sprintf("done %.0fs (%.1f s/sample)\n", as.numeric(difftime(Sys.time(), t0, units = "secs")),
            as.numeric(difftime(Sys.time(), t0, units = "secs")) / length(pick)))
write.csv(out, file.path(IO, "efm_pred.csv"), row.names = FALSE)
cat("count acc =", mean(out$efm == out$true, na.rm = TRUE), "\n")
