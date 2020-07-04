data2 = read.csv("MouldMakingData_till_27-06-2020.csv",skip = 2,header = F)
data2= data2[,1:5]
l1 = grepl("http+",data2$V1)
l2 = grepl("http+",data2$V2)
l = l1+l2
l = as.logical(l)
data2 = data2[l,]
temp = data2
l = temp$V1==""
temp[l,1] = temp[l,2]
temp$V6 = substr(temp$V1,128-11-6,128-10)
data2$V6 = temp$V6
data2$V6 = paste(substr(data2$V6,1,4),substr(data2$V6,5,6),substr(data2$V6,7,8),sep = "-")

add<-function(data,j)
{
        for(i in 1:nrow(data))
        {
                if(data[i,j]!="")
                {
                        data[i,j] = gsub("https://f002.backblazeb2.com/file/senseData/crescent/annotatedVideos",paste("/media/neo/krypton/data/data_crescent/camera_videos/original_videos",data[i,6],sep = "/"),data[i,j])
                }
        }
        data
}

data2 = add(data2,1)
data2 = add(data2,2)

data2$V3 = tolower(data2$V3)

l = data2$V3=="air gun "
data2[l,]$V3 = "air gun"

l = data2$V3=="flter placement"
data2[l,]$V3 = "filter placement"

l = data2$V3=="vent cleanin"
data2[l,]$V3 = "vent cleaning"
l = data2$V3=="vent cleaning "
data2[l,]$V3 = "vent cleaning"

data2$V2 = gsub("\n","",data2$V2)
data2$V1 = gsub("\n","",data2$V1)
write.csv(data2,"final.csv")


