std_dev<-function(data){
        sd(data)
}

avg_dev<-function(data)
{
        d1 = abs(data-mean(data))
        mean(d1)
}

rel_avg_dev<-function(data)
{
        d_bar = avg_dev(data)
        ans = (d_bar/mean(data))
        ans
}

rsd<-function(data)
{
        s = sd(data)
        ans = (s/mean(data))
        ans*100
}

s_mean<-function(data)
{
        n = length(data)
        sd(data)/(n^0.5)
}

rs_mean<-function(data)
{
        s_mean(data)/mean(data)
}