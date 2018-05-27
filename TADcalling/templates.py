# parameters are needed to be checked
hicseg_template = "library(HiCseg)\n" \
                  "matrix = as.matrix(read.table(file='{}',sep='\t',header=F,stringsAsFactors=F))\n" \
                  "dim=dim(matrix)\n" \
                  "n=dim[1]\n" \
                  "result = HiCseg_linkC_R(n,round(n/3), '{}', matrix,'D')\n" \
                  "write.table(result, 'buff_hicseg.txt', sep='\t', quote=F)\n" \
                  "results<-read.table('buff_hicseg.txt', header=T, sep='\t', stringsAsFactors=F)\n" \
                  "options(scipen=999)\n" \
                  "res<-{}\n" \
                  "results_trim<-results[results$t_hat!=0,]\n" \
                  "TADs<-data.frame('start'=(results_trim$t_hat[-(nrow(results_trim))]*res),'end'=(results_trim$t_hat[-1]*res))\n" \
                  "write.table(TADs, file='{}', quote=FALSE, sep='\t', row.names = FALSE, col.names = FALSE)\n"
