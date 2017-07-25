# Blue print for pyABC parameter inference runs.
# A proposal of how to structure an R file for usage with pyABC


#' The model to be simulated.
#' 
#' The number of parameters depends on the model and can
#' be arbitrary. However, the parameters should be real or integer numbers.
#' The return type is arbitrary as well.
model <- function(pars){
  x <- rnorm(1) + pars$x
  y <- rnorm(1) + pars$y
  c(x,y)
}

#' Calculates summary statistics from whatever the model returns
#' 
#' It is important that the summary statistics have names to store it correctly in pyABC's database
#' 
#' @param modelResult The data simulated by the model
#' @return Named list of summary statistics.
summaryStatistics <- function(modelResult){
  list(x=modelResult[1], y=modelResult[2])
}

#' Calculate distance between summary statistics
#' 
#' @param sumStatSample The summary statistics of the sample proposed py pyABC
#' @param sumStatData The summary statistics of the observed data for which we want to calculate the posterior.
#' @return A single float
distance <- function(sumStatSample, sumStatData){
  abs(sumStatSample$x - sumStatData$x) + abs(sumStatSample$y - sumStatData$y)
}

observation <- list(x=4, y=8)

# The functions model, summaryStatistics and distance have to be constructed in a
# such that the following always makes sense.
# Note: The model function can in principle already return
# a named list and the summary statistics function can just pass it through.
# A separate summary statistics function is mainly useful in a model selection
# scenario.
#print(distance(summaryStatistics(model(1,2)), summaryStatistics(model(3,4))))

