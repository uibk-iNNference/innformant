# load data
df <- read.csv("../boundaries/i7-9700/boundary-experiments-7.csv")
df$identified_MA <- factor(df$identified_MA)
df$total_steps <- df$local_steps + df$remote_steps

df$s0_label <- factor(df$s0_label)
df$s1_label <- factor(df$s1_label)
df$s2_label <- factor(df$s2_label)
df$s3_label <- factor(df$s3_label)
df$s4_label <- factor(df$s4_label)
df$s5_label <- factor(df$s5_label)
df$s6_label <- factor(df$s6_label)

df <- df[df$identified_MA != "failure",]

fmnist <- df[df$model_type == "fmnist",]

cifar <- df[df$model_type == "cifar10",]

nsamples <- 400

# success_rate
fmnist_success_rate <- nrow(fmnist)/nsamples
fmnist_fail_rate <- 1-fmnist_success_rate
paste("FMNIST Success rate: ", fmnist_success_rate)

cifar_success_rate <- nrow(cifar)/nsamples
cifar_fail_rate <- 1-cifar_success_rate
paste("CIFAR Success rate: ", cifar_success_rate)

# distribution of classified architectures
paste("FMNIST")
tapply(fmnist$total_psnr, list(fmnist$identified_MA), function(v) length(v)/nrow(fmnist))

paste("CIFAR")
tapply(cifar$total_psnr, list(cifar$identified_MA), function(v) length(v)/nrow(cifar))

# psnr in general
cat("FMNIST")
summary(fmnist$total_psnr)
# plot(density(fmnist$total_psnr))

# max PSNR for each identifiable MA
cat("Max:")
tapply(fmnist$total_psnr, list(fmnist$identified_MA), max)
cat("Min:")
tapply(fmnist$total_psnr, list(fmnist$identified_MA), min)

cat("\n\n")

cat ("CIFAR")
summary(cifar$total_psnr)
# plot(density(fmnist$total_psnr))

# max PSNR for each identifiable MA
cat("Max:")
tapply(cifar$total_psnr, list(cifar$identified_MA), max)
cat("Min:")
tapply(cifar$total_psnr, list(cifar$identified_MA), min)


# # TODO: remove outliers
len_before <- nrow(fmnist)
fmnist <- fmnist[fmnist$local_steps < 500 & fmnist$remote_steps < 500 & fmnist$total_steps < 900,]
cat("Difference FMNIST: ", len_before - nrow(fmnist))
nrow(fmnist)
len_before <- nrow(cifar)
cifar <- cifar[cifar$local_steps < 500 & cifar$remote_steps < 500 & cifar$total_steps < 900,]
cat("Difference CIFAR: ", len_before - nrow(cifar))


generate_box_macros <- function(box, outfile){
    cat(paste(c(
        paste("\\providecommand{\\outliers}{", paste(round(box$out,4),collapse=","), "}"),
        paste("\\providecommand{\\lowerwhisker}{", round(box$stats[1], 4), "}"),
        paste("\\providecommand{\\qone}{", round(box$stats[2], 4), "}"),
        paste("\\providecommand{\\median}{", round(box$stats[3], 4), "}"),
        paste("\\providecommand{\\qthree}{", round(box$stats[4], 4), "}"),
        paste("\\providecommand{\\upperwhisker}{", round(box$stats[5], 4), "}")
        ), collapse="\n"), file=outfile)
}

# distribution of required local and remote steps
dir.create("generated")
lims = c(0,500)
par(mfrow=c(1,4))
res <- boxplot(fmnist$local_steps, ylim=lims,las=1)
generate_box_macros(res, "generated/fmnist-local-steps.tex")
res <- boxplot(fmnist$remote_steps, ylim=lims, yaxt='n')
generate_box_macros(res, "generated/fmnist-remote-steps.tex")

res <- boxplot(cifar$local_steps, ylim=lims, yaxt='n')
generate_box_macros(res, "generated/cifar-local-steps.tex")
res <- boxplot(cifar$remote_steps, ylim=lims, yaxt='n')
generate_box_macros(res, "generated/cifar-remote-steps.tex")

# psnr over steps
line_xs <- data.frame(total_steps = c(-500,900), local_steps = c(-500,500))


dir.create("macros")
par(mfrow=c(2,2))
plot(fmnist$total_steps,fmnist$total_psnr)
cat(paste("\\draw (", round(fmnist$total_steps, 4), ",", round(fmnist$total_psnr, 4), ") node[point] {};", collapse="\n"), file="macros/fmnist-psnr-steps.tex")
fit <- lm(formula = total_psnr~ total_steps, data=fmnist)
abline(fit$coef, col=2)
cat("\\draw ", paste("(", line_xs$total_steps, ", ", predict(fit, line_xs), ")", collapse=" -- "), ";\n")

plot(cifar$total_steps,cifar$total_psnr)
cat(paste("\\draw (", round(cifar$total_steps, 4), ",", round(cifar$total_psnr, 4), ") node[point] {};", collapse="\n"), file="macros/cifar-psnr-steps.tex")
fit <- lm(formula = total_psnr~ total_steps, data=cifar)
abline(fit$coef, col=2)
cat("\\draw ", paste("(", line_xs$total_steps, ", ", predict(fit, line_xs), ")", collapse=" -- "), ";\n")

# remote steps over local steps
plot(fmnist$local_steps,fmnist$remote_steps)
cat(paste("\\draw (", round(fmnist$local_steps, 4), ",", round(fmnist$remote_steps, 4), ") node[point] {};", collapse="\n"), file="macros/fmnist-remote-local.tex")
fit <- lm(remote_steps~local_steps, data=fmnist)
abline(fit$coef, col=2)
cat("\\draw ", paste("(", line_xs$local_steps, ", ", predict(fit, line_xs), ")", collapse=" -- "), ";\n")

plot(cifar$local_steps,cifar$remote_steps)
cat(paste("\\draw (", round(cifar$local_steps, 4), ",", round(cifar$remote_steps, 4), ") node[point] {};", collapse="\n"), file="macros/cifar-remote-local.tex")
fit <- lm(formula = remote_steps~local_steps, data=cifar)
abline(fit$coef, col=2)
cat("\\draw ", paste("(", line_xs$local_steps, ", ", predict(fit, line_xs), ")", collapse=" -- "), ";\n")

# which labels are isolated and which are contrast
# get unique labels for each row
get_identifiables <- function(label_row){
    uniques <- sapply(label_row,unique)
    return(uniques[which.min(sapply(uniques, function(x) sum(x==label_row)))])
}
get_contrast <- function(label_row){
    uniques <- sapply(label_row,unique)
    return(uniques[which.max(sapply(uniques, function(x) sum(x==label_row)))])
}
print_confusion_line <- function(l) paste(lapply(l, function(x) paste("\\confusionCell{",x,"}{",round(x/tmax*100,0),"}%")), collapse='\n')

M <- cbind(fmnist$s0_label, fmnist$s1_label, fmnist$s2_label, fmnist$s3_label, fmnist$s4_label, fmnist$s5_label, fmnist$s6_label)
M[1,]
fmnist$identifiables <- factor(apply(M, 1, get_identifiables))
fmnist$contrast <- factor(apply(M, 1, get_contrast))

M <- cbind(cifar$s0_label, cifar$s1_label, cifar$s2_label, cifar$s3_label, cifar$s4_label, cifar$s5_label, cifar$s6_label)
cifar$identifiables <- factor(apply(M, 1, get_identifiables))
cifar$contrast <- factor(apply(M, 1, get_contrast))

# TODO: confusion matrix
tbl <- table(fmnist$identifiables, fmnist$contrast)
tmax <- max(tbl)

cat(paste(apply(tbl,1, print_confusion_line), collapse="\n\n"), file="macros/fmnist-confusion.tex")

tbl <- table(cifar$identifiables, cifar$contrast)
tmax <- max(tbl)

cat(paste(apply(tbl,1, print_confusion_line), collapse="\n\n"), file="macros/cifar-confusion.tex")

fmnist_labels <- c(4, 4, 9, 7, 5, 1, 0, 5, 7, 4, 0, 8, 2, 3, 9, 0, 7, 7, 2, 2, 0, 4,
                    4, 4, 2, 7, 7, 4, 2, 4, 7, 5, 9, 5, 4, 4, 3, 3, 1, 7, 5, 3, 0, 0,
                    0, 6, 9, 9, 7, 6, 2, 0, 0, 9, 6, 1, 5, 7, 0, 2, 1, 9, 7, 2, 4, 6,
                    2, 2, 6, 5, 1, 5, 6, 2, 3, 6, 4, 6, 4, 6, 4, 5, 8, 9, 2, 5, 1, 9,
                    9, 1, 4, 1, 0, 5, 8, 0, 4, 2, 0, 0, 1, 4, 6, 4, 2, 5, 5, 9, 4, 4,
                    5, 6, 3, 7, 2, 4, 7, 9, 5, 7, 2, 1, 3, 1, 0, 4, 4, 9, 0, 3, 5, 0,
                    7, 1, 7, 1, 0, 3, 3, 8, 5, 2, 2, 7, 1, 7, 4, 4, 2, 0, 0, 1, 6, 0,
                    1, 0, 3, 5, 1, 8, 0, 5, 9, 3, 3, 7, 1, 3, 7, 9, 9, 2, 1, 4, 6, 2,
                    4, 0, 7, 9, 5, 0, 6, 9, 5, 7, 1, 2, 6, 3, 1, 0, 0, 2, 9, 0, 8, 3,
                    9, 8, 5, 6, 3, 5, 2, 3, 5, 3, 2, 7, 4, 8, 8, 5, 2, 4, 0, 6, 2, 4,
                    6, 2, 3, 5, 7, 2, 2, 9, 2, 6, 3, 5, 9, 8, 2, 9, 1, 8, 0, 6, 4, 7,
                    1, 3, 6, 4, 3, 8, 1, 2, 9, 9, 7, 3, 3, 2, 1, 1, 6, 6, 5, 4, 2, 4,
                    2, 7, 7, 3, 6, 3, 4, 9, 1, 4, 4, 1, 9, 3, 7, 7, 7, 4, 9, 8, 0, 6,
                    7, 4, 0, 2, 2, 3, 1, 2, 3, 1, 1, 6, 4, 3, 0, 2, 1, 5, 6, 8, 5, 4,
                    2, 2, 7, 8, 2, 1, 3, 3, 7, 5, 7, 7, 2, 7, 3, 8, 9, 1, 7, 1, 2, 7,
                    2, 5, 1, 9, 6, 7, 7, 0, 3, 9, 0, 4, 9, 3, 8, 4, 7, 4, 1, 9, 8, 1,
                    8, 3, 0, 1, 2, 3, 8, 5, 7, 6, 8, 2, 9, 1, 6, 2, 1, 3, 9, 5, 4, 2,
                    4, 7, 0, 7, 0, 3, 5, 4, 1, 8, 6, 7, 6, 4, 4, 7, 5, 9, 8, 1, 5, 3,
                    7, 7, 2, 9)

cifar_labels <- c(7, 0, 6, 9, 5, 1, 7, 0, 3, 2, 7, 2, 7, 9, 6, 8, 8, 8, 7, 9, 4, 2,
                   3, 5, 0, 0, 2, 8, 1, 8, 2, 0, 7, 2, 0, 0, 9, 1, 0, 5, 1, 5, 0, 3,
                   5, 0, 2, 0, 5, 3, 7, 4, 1, 4, 2, 5, 1, 5, 2, 0, 7, 2, 2, 1, 8, 5,
                   3, 5, 3, 0, 6, 1, 5, 2, 8, 2, 5, 3, 0, 8, 8, 8, 6, 5, 8, 1, 9, 8,
                   1, 3, 6, 0, 8, 7, 5, 3, 1, 8, 1, 4, 0, 8, 7, 3, 4, 1, 4, 3, 2, 2,
                   1, 4, 7, 4, 9, 3, 4, 4, 2, 8, 1, 2, 8, 5, 2, 4, 0, 5, 9, 1, 5, 1,
                   0, 2, 7, 7, 1, 0, 5, 4, 6, 9, 7, 6, 0, 4, 0, 5, 1, 2, 7, 0, 6, 4,
                   7, 5, 5, 4, 7, 8, 1, 8, 2, 4, 9, 3, 1, 1, 4, 9, 9, 3, 7, 8, 2, 6,
                   0, 3, 4, 2, 0, 0, 2, 2, 5, 7, 1, 3, 0, 2, 0, 0, 7, 8, 6, 2, 8, 5,
                   0, 5, 3, 4, 1, 7, 9, 5, 5, 8, 4, 9, 3, 7, 9, 1, 9, 5, 3, 3, 0, 0,
                   7, 8, 0, 2, 9, 9, 2, 9, 2, 6, 1, 6, 0, 0, 1, 4, 3, 1, 7, 1, 1, 8,
                   8, 2, 2, 5, 4, 0, 8, 7, 3, 6, 4, 7, 1, 8, 4, 0, 0, 4, 1, 7, 0, 9,
                   2, 4, 2, 3, 4, 2, 5, 4, 1, 4, 7, 9, 4, 2, 5, 5, 8, 8, 5, 4, 9, 5,
                   0, 1, 9, 9, 8, 7, 0, 7, 0, 7, 6, 4, 6, 6, 4, 9, 5, 1, 8, 3, 0, 5,
                   2, 7, 1, 7, 8, 2, 5, 4, 3, 1, 7, 6, 6, 3, 9, 9, 9, 9, 9, 6, 6, 5,
                   5, 0, 4, 0, 1, 7, 8, 1, 1, 8, 1, 9, 3, 6, 3, 3, 4, 3, 9, 2, 6, 2,
                   4, 6, 9, 9, 1, 3, 9, 5, 6, 8, 7, 4, 2, 9, 3, 4, 0, 5, 9, 3, 7, 2,
                   3, 5, 2, 5, 6, 8, 2, 2, 4, 5, 1, 8, 0, 3, 3, 0, 1, 7, 1, 8, 3, 9,
                   2, 8, 5, 7)

# re-add outliers
fmnist <- df[df$model_type == "fmnist",]

cifar <- df[df$model_type == "cifar10",]

# get desired labels
fmnist$gt_label <- fmnist_labels[fmnist$sample_index]
cifar$gt_label <- fmnist_labels[cifar$sample_index]
combined <- rbind(fmnist,cifar)

# save to new csv
write.csv(combined, "including_ground_truth.csv")
