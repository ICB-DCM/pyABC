args = commandArgs(trailingOnly=TRUE)
pars = list()
names_ = list()
for (arg in args) {
    s = strsplit(arg, "=")
    names_ = c(names_, s[[1]][1])
    pars = c(pars, s[[1]][2])
}

names(pars) = names_


sumstat <- function(pars){
    model_result = readRDS(pars$model_output)
    sumstat = list(
        x=model_result[1], y=model_result[2],
	mtcars=mtcars,  # Can also pass data frames
        cars=cars,
        arbitraryKey="Some random text")
    # print(sumstat)
    saveRDS(sumstat, pars$target)
}


sumstat(pars)
