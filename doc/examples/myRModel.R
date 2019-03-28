# Blue print for pyABC parameter inference runs.
# A proposal of how to structure an R file for usage with pyABC


#' The model to be simulated.
#' In this example, it is just a multivariate normal
#' distribution. The number of parameters depends on the
#' model and can be arbitrary. However, the parameters
#' should be real numbers.
#' The return type is arbitrary as well.
myModel <- function(pars){
  x <- rnorm(1) + pars$meanX
  y <- rnorm(1) + pars$meanY
  c(x,y)  # It is not important that it is a vector.
}

#' Calculates summary statistics from whatever the model returns
#' 
#' It is important that the summary statistics have names
#' to store them correctly in pyABC's database.
#' In many cases, the summary statistics function might just
#' pass through the result of the model function if the
#' summary statistics calculation is already
#' done there. Splitting summary statistics and the model
#' makes most sense in a model selection scenario.
#' 
#' @param modelResult The data simulated by the model
#' @return Named list of summary statistics.
mySummaryStatistics <- function(modelResult){
  list(x=modelResult[1],
       y=modelResult[2],
       mtcars=mtcars,  # Can also pass data frames
       cars=cars,
       arbitraryKey="Some random text")
}

#' Calculate distance between summary statistics
#' 
#' @param sumStatSample The summary statistics of the sample
#' proposed py pyABC
#' @param sumStatData The summary statistics of the observed
#'        data for which we want to calculate the posterior.
#' @return A single float
myDistance <- function(sumStatSample, sumStatData){
  sqrt((sumStatSample$x - sumStatData$x)^2
       + abs(sumStatSample$y - sumStatData$y)^2)
}


# We store the observed data as named list
# in a variable.
# This is passed by pyABC as to myDistance
# as the sumStatData argument
mySumStatData <- list(x=4, y=8, mtcars=mtcars, cars=cars)

# The functions for the model, the summary
# statistics and distance
# have to be constructed in
# such a way that the following always makes sense:
#
# myDistance(
#   mySummaryStatistics(myModel(list(meanX=1, meanY=2))),
#   mySummaryStatistics(myModel(list(meanX=2, meanY=2))))

