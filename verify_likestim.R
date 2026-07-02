suppressMessages(library(forensim))
# Genuineness check: on the textbook strusa example, does likestim differ from mincontri?
data(strusa)
set.seed(1)
for (k in 2:5) {
  gen <- simugeno(strusa, n = c(0, 0, 50))
  mk <- simumix(gen, ncontri = c(0, 0, k))
  L <- as.integer(likestim(mk, strusa, refpop = "Hisp")[1, "max"])
  M <- as.integer(mincontri(mk))
  cat(sprintf("true k=%d : likestim=%d  mincontri(MAC)=%d\n", k, L, M))
}
