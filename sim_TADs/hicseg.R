install.packages("HiCseg")
library(HiCseg)
noise_values <- c(4, 8, 12, 16, 20)
sim_values <- seq(1, 5, 1)
for (noise in noise_values) {
  for (sim in sim_values) {
    matrix = as.matrix(read.table(file=paste("matrices/simHiC_countMatrix_noise", noise, "_sim", sim, ".txt.gz", sep=""),sep="\t",header=F,stringsAsFactors=F))
    dim=dim(matrix)
    n=dim[1]
    result = HiCseg_linkC_R(n,round(n/3),"G",matrix,"D")
    write.table(result, "buff_hicseg.txt", sep="\t", quote=F)
    results<-read.table("buff_hicseg.txt",header=T,sep="\t",stringsAsFactors=F)
    options(scipen=999)
    res<-40000
    results_trim<-results[results$t_hat!=0,]
    TADs<-data.frame("start"=(results_trim$t_hat[-(nrow(results_trim))]*res),"end"=(results_trim$t_hat[-1]*res))
    write.table(TADs, file=paste("yielded/hicseg_noise", noise, "_sim", sim, ".txt", sep=""), quote=FALSE, sep="\t", row.names = FALSE, col.names = FALSE)
  }
}
