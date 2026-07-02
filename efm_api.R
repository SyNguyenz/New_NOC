suppressMessages(library(euroformix))
fns <- ls("package:euroformix")
cat("=== NOC / MLE / search / data functions ===\n")
print(grep("noc|NOC|MLE|search|Search|contLik|prepare|Qassig|sample_|tableR|getData|calcMLE|freq", fns, value=TRUE))
cat("\n=== contLikSearch args ===\n"); if(exists("contLikSearch")) print(args(contLikSearch))
cat("\n=== contLikMLE args ===\n"); if(exists("contLikMLE")) print(args(contLikMLE))
cat("\n=== calcMLE args (if any) ===\n"); if(exists("calcMLE")) print(args(calcMLE))
cat("\n=== prepareData / Qassignate args ===\n")
for(f in c("prepareData","Qassignate","sample_tableToList","freqImport")) if(exists(f)){cat("--",f,"\n");print(args(get(f)))}
