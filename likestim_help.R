suppressMessages(library(forensim))
# print the example + structure expected by likestim / tabfreq / simumix
cat("=== likestim help (Description + Examples) ===\n")
db <- tools::Rd_db("forensim")
show <- function(name){
  rd <- db[[paste0(name,".Rd")]]
  if(is.null(rd)) { cat("(no Rd for",name,")\n"); return() }
  txt <- paste(capture.output(tools::Rd2txt(rd)), collapse="\n")
  cat(txt, "\n\n")
}
show("likestim")
cat("=========== tabfreq ===========\n"); show("tabfreq")
