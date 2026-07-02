repos <- "https://cloud.r-project.org"
# binary deps first (avoid compiling these)
need <- c("remotes", "Rcpp", "RcppArmadillo", "cubature", "gtools", "MASS", "numDeriv")
for (p in need) if (!requireNamespace(p, quietly = TRUE)) {
  cat("installing", p, "...\n"); install.packages(p, repos = repos)
}
cat("deps ready. Installing euroformix from CRAN archive (compiles C++) ...\n")
ok <- tryCatch({
  remotes::install_version("euroformix", repos = repos, upgrade = "never", quiet = FALSE)
  TRUE
}, error = function(e) { cat("install_version FAILED:", conditionMessage(e), "\n"); FALSE })
if (!ok) {
  cat("Trying GitHub oyvble/euroformix ...\n")
  tryCatch(remotes::install_github("oyvble/euroformix", upgrade = "never", quiet = FALSE),
           error = function(e) cat("GitHub FAILED:", conditionMessage(e), "\n"))
}
cat("euroformix installed:", "euroformix" %in% rownames(installed.packages()), "\n")
if ("euroformix" %in% rownames(installed.packages()))
  cat("version:", as.character(packageVersion("euroformix")), "\n")
