sumstat <- function(model_output){
    model_result = read.csv(model_output$loc)
    # print(model_result)
    sumstat = list(
        loc=model_output$loc,
        x=sum(model_result$s0),
        y=sum(model_result$s1)
    )
}
