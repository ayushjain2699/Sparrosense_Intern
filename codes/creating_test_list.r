
create<-function(data2){
        data3 <- data.table(va=character(), vb=character(), vc=character())
        for (i in 1:nrow(data2))
        {
                for (j in 1: 7)
                {
                        row = data.table(va = data2[i,1],vb = as.character(1+16*(j-1)),vc = data2[i,2])
                        data3 = rbind(data3,row)
                }
        }
        data3
}