have <- "euroformix" %in% rownames(installed.packages())
cat("euroformix installed:", have, "\n")
cat("R version:", as.character(getRversion()), "\n")
cat("lib path:", .libPaths()[1], "\n")
if (have) {
  suppressMessages(library(euroformix))
  cat("euroformix version:", as.character(packageVersion("euroformix")), "\n")
}
