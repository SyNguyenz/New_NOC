suppressMessages(library(forensim))
args <- commandArgs(trailingOnly = TRUE)
mode <- ifelse(length(args) >= 1, args[1], "subset")   # "subset" or "full"
IO <- "C:/Tailieu/TinSinh/Project/new_NOC/forensim_io"

# ---- build tabfreq from freq.csv ----
freq <- read.csv(file.path(IO, "freq.csv"), colClasses = c("character", "character", "numeric"))
loci <- unique(freq$locus)
tablist <- list()
for (loc in loci) {
  s <- freq[freq$locus == loc, ]
  v <- s$freq; names(v) <- s$allele
  tablist[[loc]] <- v
}
tabF <- new("tabfreq", tab = list(pop = tablist), which.loc = loci, pop.names = factor("pop"))
cat("tabfreq built:", length(loci), "loci; valid:", is.tabfreq(tabF), "\n")

# ---- read mixtures (long: sample, locus, allele) ----
mix <- read.csv(file.path(IO, "mix_test.csv"), colClasses = c("integer", "character", "character"))
trueN <- as.integer(read.csv(file.path(IO, "noc_test.csv"))$noc)   # index = sample+1

build_mix <- function(rows) {
  al <- list()
  for (loc in loci) {
    a <- rows$allele[rows$locus == loc]
    a <- a[a %in% names(tablist[[loc]])]                # keep only alleles present in freq
    if (length(a) > 0) al[[loc]] <- unique(a)
  }
  wl <- names(al)
  new("simumix", ncontri = 1L, mix.prof = matrix(NA_character_, nrow = length(wl), ncol = 1),
      mix.all = al, which.loc = wl, popinfo = factor("pop"))
}

samples <- sort(unique(mix$sample))
if (mode == "subset") {
  # stratified: up to 15 per true NOC
  pick <- integer(0)
  for (k in 1:5) pick <- c(pick, head(samples[trueN[samples + 1] == k], 15))
  samples <- pick
}
cat("running likestim on", length(samples), "samples (mode =", mode, ") ...\n")

t0 <- Sys.time()
res <- data.frame(sample = samples, true = trueN[samples + 1], like = NA_integer_, mac = NA_integer_)
for (ii in seq_along(samples)) {
  s <- samples[ii]
  m <- build_mix(mix[mix$sample == s, ])
  res$like[ii] <- tryCatch(as.integer(likestim(m, tabF)[1, "max"]), error = function(e) NA)
  res$mac[ii]  <- tryCatch(as.integer(mincontri(m)), error = function(e) NA)
}
dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("done %.0fs (%.0f ms/sample)\n", dt, dt / length(samples) * 1000))

acc <- function(p, t) mean(p == t, na.rm = TRUE)
cat(sprintf("\nlikestim count acc = %.3f | mincontri(MAC) acc = %.3f\n",
            acc(res$like, res$true), acc(res$mac, res$true)))
cat("\nper-NOC likestim accuracy:\n")
for (k in 1:5) {
  m <- res$true == k
  if (sum(m)) cat(sprintf("  NOC%d: like=%.3f  mac=%.3f  (n=%d)  pred=%s\n", k,
      acc(res$like[m], k), acc(res$mac[m], k), sum(m),
      paste(sort(table(res$like[m]), decreasing = TRUE), collapse = ",")))
}
write.csv(res, file.path(IO, paste0("forensim_pred_", mode, ".csv")), row.names = FALSE)
cat("\nsaved ->", file.path(IO, paste0("forensim_pred_", mode, ".csv")), "\n")
