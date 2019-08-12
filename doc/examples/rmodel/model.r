args = commandArgs(trailingOnly=TRUE)
pars = list()
names_ = list()
for (arg in args) {
    s = strsplit(arg, "=")
    names_ = c(names_, s[[1]][1])
    pars = c(pars, s[[1]][2])
}
names(pars) = names_


model <- function(pars){
  x <- rnorm(1) + as.numeric(pars$meanX)
  y <- rnorm(1) + as.numeric(pars$meanY)
  out <- c(x, y)  # It is not important that it is a vector.

  # now this must be written to file
  saveRDS(out, file=pars$target)
}


model(pars)
