suppressMessages(library(forensim))
cat("=== exported functions (NOC-related) ===\n")
fns <- ls("package:forensim")
print(grep("noc|contri|nb|like|lik|Pnm|simu|maxlik|A$", fns, value=TRUE, ignore.case=TRUE))
cat("\n=== all functions ===\n"); print(fns)
cat("\n=== bundled datasets ===\n")
print(data(package="forensim")$results[, "Item"])
cat("\n=== help for likelihood NOC estimator (if any) ===\n")
for (f in c("maxlik","likestim","nbcont","Pnm","simumix")) {
  if (exists(f)) { cat("--", f, "args:\n"); print(args(get(f))) }
}
