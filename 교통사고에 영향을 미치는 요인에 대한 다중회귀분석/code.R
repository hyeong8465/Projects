library(readxl)
library(car)

A=read_excel('data.xlsx',col_name = TRUE)
head(A)
as.data.frame(A)
summary(A$accident)
summary(A$car)
summary(A$population)
summary(A$road)
summary(A$safety)
summary(A$rain)
sd(A$accident)
sd(A$car)
sd(A$population)
sd(A$road)
sd(A$safety)
sd(A$rain)

par(mfrow = c(2,3))
plot(A$year,A$accident,type="l")
plot(A$year,A$car,type="l")
plot(A$year,A$population,type="l")
plot(A$year,A$road,type="l")
plot(A$year,A$safety,type="l")
plot(A$year,A$rain,type="l")
plot(A, cex.lab=3)

lm.A=lm(accident ~ car + population + road + safety + rain, data = A)
summary(lm.A)
vif(lm.A)


par(mfrow = c(3,1))
qqnorm(rstandard(lm.A))
abline(0,1,col=1)
plot(hatvalues(lm.A), rstandard(lm.A))
plot(rstandard(lm.A),ylim=c(-3,3))

library(lmtest)
dwtest(lm.A)

plot(A[,c(2:6)])
cormat = cor(A[,c(2:7)])

library(corrplot) 
par(mfrow = c(1,1))
corrplot(cormat, method = "color", col = colorRampPalette(c("blue", "white", "red"))(200), 
         type = "upper", order = "hclust", 
         addCoef.col = "black", # 상관계수 표시
         tl.col = "black", tl.srt = 45, # 텍스트 색상 및 회전
         diag = FALSE) # 대각선 생략

lm.A2=step(lm.A,direction = "backward")
summary(lm.A2)
vif(lm.A2)



