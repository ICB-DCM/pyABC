args = commandArgs(trailingOnly=TRUE)
pars = list()
names_ = list()
for (arg in args) {
    s = strsplit(arg, "=")
    names_ = c(names_, s[[1]][1])
    pars = c(pars, s[[1]][2])
}

names(pars) = names_


distance <- function(pars){
    sumstat_0 = readRDS(pars$sumstat_0)
    # insert observed data here directly
    # note: by design, pyabc can currently not completely make sure
    # that the observed data are always the second argument,
    # and the simulated ones the first
    sumstat_1 = list(x=4, y=8, mtcars=mtcars, cars=cars)

    dist = sqrt((sumstat_0$x - sumstat_1$x)^2
        + abs(sumstat_0$y - sumstat_1$y)^2)

    write(dist, pars$target)
}


distance(pars)
