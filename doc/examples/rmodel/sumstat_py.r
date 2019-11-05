sumstat <- function(model_output){
    model_result = read.csv(model_output$loc)
    print(model_result)
    sumstat = list(
        x=model_result$s0, y=model_result$s1,
	mtcars=mtcars,  # Can also pass data frames
        cars=cars,
        arbitraryKey="Some random text")    
}
