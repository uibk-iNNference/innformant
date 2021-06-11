# RB 02/21
# setwd("~/Documents/Projekte/IHMMSEC2021/data/")

df <- read.csv("boundary-experiments-7.csv")

df$success <- df$success=="True"
df$model_type <- factor(df$model_type)
df$total_steps <- df$local_steps + df$remote_steps


sel <- df$model_type=="fmnist" & df$success

tapply(df$total_psnr,list(df$model_type,df$success),length)

hist(df$local_steps[sel])

plot(df$local_steps[sel], df$remote_steps[sel])


plot(df$local_steps[sel], df$local_psnr[sel], ylim=c(30,170))
abline(lm(df$local_psnr~df$local_steps,subset=sel)$coef,col=2)

plot(df$remote_steps[sel], df$remote_psnr[sel], ylim=c(30,170))
abline(lm(df$remote_psnr~df$remote_steps,subset=sel)$coef,col=2)

plot(df$total_steps[sel], df$total_psnr[sel], ylim=c(30,170))
abline(lm(df$total_psnr~df$total_steps,subset=sel)$coef,col=2)




M <- cbind(df$s0_label, df$s1_label, df$s2_label, df$s3_label, df$s4_label, df$s5_label, df$s6_label)

# how many partitions?
table(apply(M,1,function(x) length(table(x))),df$success)  # never more than two (if success)

apply(M[df$success,],1, function(x) { w <- which(table(factor(x,levels=0:9))==1)-1 ; which(x==w) })  # which server is identifiable?

apply(M[df$success,],1, function(x) as.numeric(which(table(factor(x,levels=0:9))==1))-1 ) # which label is isolated?

apply(M[df$success,],1, function(x) as.numeric(which(table(factor(x,levels=0:9))>1))-1 ) # which label is the contrast?

plot((df$s6_conf2-df$s5_conf2)[df$success])

