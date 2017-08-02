install.packages("feather")
install.packages("jsonlite")

library("feather")
library("jsonlite")

loadedDf <- data.frame(feather("df.feather"))

jsonStr <- loadedDf$sumstat_ss_df[1]

sumStatDf <- fromJSON(jsonStr)
