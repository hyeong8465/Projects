# 데이터
data=read.csv("data.csv")
str(data)
table(data$AlcoholDrinking)
# 수면시간 변수 변명 6~10일때 0 / 적정수면시간 부족 혹은 초과시 1
data$SleepTime = ifelse(data$SleepTime <6 , 1,
                     ifelse(data$SleepTime >10 , 1 , 0))

# BMI, 인종, 건강상태, 나이에 대한 더미변수 생성
library(fastDummies)
df=dummy_columns(.data=data, select_columns = c('BMI','Race','GenHealth','AgeCategory')) 
str(df)
library(reshape)
df = rename(df,c("Race_American Indian/Alaskan Native" = "Race_IndianAlaskanNative")) # 변수명 변경

# 로지스틱 회귀분석
# 1번모델 BMI, 인종, GenHealth, AgeCategory 더미변수/ 신체건강, 정신건강 변수제거
glm.df1 = glm(HeartDisease~ Smoking + AlcoholDrinking + Stroke + DiffWalking + Sex + Diabetic + PhysicalActivity + SleepTime + Asthma + KidneyDisease + SkinCancer
              + BMI_1 + BMI_3 + BMI_4 + BMI_5
              + Race_IndianAlaskanNative + Race_Asian+ Race_Black + Race_Hispanic + Race_Other
              + GenHealth_1 + GenHealth_2 + GenHealth_3 + GenHealth_4
              + AgeCategory_1 + AgeCategory_2 + AgeCategory_3 + AgeCategory_4 + AgeCategory_5 + AgeCategory_6 + AgeCategory_7 + AgeCategory_8
              + AgeCategory_9 + AgeCategory_10 + AgeCategory_11 + AgeCategory_12, data=df, family=binomial(link='logit'))
summary(glm.df1)
# 각각 BMI(2), 인종(백인), GenHealth(0), 나이(0) baseline
# 과음 변수가 음의 기울기(예상과 다른 결과), 신체활동 유의미하지 않음(제거), 인종 기타, 원주민/알래스카 유의미하지 않음
# BMI 1그룹도 유의하지 않음 -> 2그룹이랑 합쳐서 baseline으로
# 나이 1그룹이 유의미하지 않지만 계수가 비슷한 정도로 커지므로 연속형으로 처리
# GenHealth도 마찬가지

BMI_12 = df$BMI_1 + df$BMI_2 # 저체중, 정상그룹 합한 그룹
df2 = cbind(df,BMI_12)
str(df2)

#2번 모델 
#PhysicalHealth 제거, GenHealth, AgeCategory 연속형 처리, BMI 01이 baseline
glm.df2 = glm(HeartDisease~ Smoking + AlcoholDrinking + Stroke + DiffWalking + Sex + Diabetic + SleepTime + Asthma + KidneyDisease + SkinCancer
              + BMI_3 + BMI_4 + BMI_5
              + Race_IndianAlaskanNative + Race_Asian+ Race_Black + Race_Hispanic + Race_Other
              + GenHealth + AgeCategory, data=df2, family=binomial(link='logit'))
summary(glm.df2)
#BMI와 인종만 더미처리, 여전히 과음은 음의 기울기, 인종 기타, 원주민/알래스카 무의미

#3번 모델
# 과음에 대한 교호작용이 있을 것 같은 변수(BMI, 성별, 흡연, 나이, 당뇨, 뇌졸중) 모델에 추가
# AgeCategory*Sex 교호작용 추가
glm.df3 = glm(HeartDisease~ Smoking + AlcoholDrinking + Stroke + DiffWalking + Sex + Diabetic + SleepTime + Asthma + KidneyDisease + SkinCancer
              + BMI_3 + BMI_4 + BMI_5
              + Race_IndianAlaskanNative + Race_Asian+ Race_Black + Race_Hispanic + Race_Other
              + GenHealth + AgeCategory
              + AlcoholDrinking*BMI_3 + AlcoholDrinking*BMI_4 + AlcoholDrinking*BMI_5
              + AlcoholDrinking*Sex + AlcoholDrinking*Smoking + AlcoholDrinking*AgeCategory
              + AlcoholDrinking*Diabetic + AlcoholDrinking*Stroke
              + AgeCategory*Sex,data=df2, family=binomial(link='logit'))
summary(glm.df3)
# 뇌졸중만 교호작용이 유의미, 과음변수 자체의 유의성은 떨어졌음
# AgeCategory*Sex 교호작용 유의미

#4번 모델
#유의미한 교호작용과 인종 변수만 남김
glm.df4 = glm(HeartDisease~ Smoking + AlcoholDrinking + Stroke + DiffWalking + Sex + Diabetic + SleepTime + Asthma + KidneyDisease + SkinCancer
              + BMI_3 + BMI_4 + BMI_5
              + Race_Asian+ Race_Black + Race_Hispanic
              + GenHealth + AgeCategory
              + AlcoholDrinking*Stroke + AgeCategory*Sex,data=df2, family=binomial(link='logit'))
summary(glm.df4)

library(car)
vif(glm.df4) # 다중공선성 체크 
round(exp(glm.df4$coefficients),2) # 오즈비

#5번 모델
#AgeCategory*Sex 대신 AlcoholDrinking*Sex 추가
glm.df5 = glm(HeartDisease~ Smoking + AlcoholDrinking + Stroke + DiffWalking + Sex + Diabetic + SleepTime + Asthma + KidneyDisease + SkinCancer
              + BMI_3 + BMI_4 + BMI_5
              + Race_Asian+ Race_Black + Race_Hispanic
              + GenHealth + AgeCategory
              + AlcoholDrinking*Stroke + AlcoholDrinking*Sex,data=df2, family=binomial(link='logit'))
summary(glm.df5)

vif(glm.df5) # 다중공선성 체크 
round(exp(glm.df5$coefficients),2) # 오즈비

#변수 선택법
glm.df5.b=step(glm.df5, direction = 'backward') # 후진제거
summary(glm.df5.b)
glm.df5.f=step(glm.df5, direction = 'forward') # 전진선택
summary(glm.df5.f)

#적합도
library(DescTools)
PseudoR2(glm.df5, which = 'all') # PseudoR2 적합도 계산









glm.df7 = glm(HeartDisease~ Smoking + AlcoholDrinking + Stroke + DiffWalking + Sex + Diabetic + SleepTime + Asthma + KidneyDisease + SkinCancer
              + BMI_3 + BMI_4
              + Race_Asian+ Race_Black + Race_Hispanic
              + GenHealth + AgeCategory
              + AlcoholDrinking*Stroke + AlcoholDrinking*Sex,data=df2, family=binomial(link='logit'))
summary(glm.df7)


# 과음과 관련된 변수 시각화
par(mfrow=c(2,2))
barplot(table(df$AlcoholDrinking , df$BMI), xlab = 'BMI', legend = c('AlcoholDrinking x', 'AlcoholDrinking o'))
barplot(table(df$AlcoholDrinking , df$Sex), xlab = 'Sex', legend = c('AlcoholDrinking x', 'AlcoholDrinking o'))
barplot(table(df$AlcoholDrinking , df$Stroke), xlab = 'Stroke', legend = c('AlcoholDrinking x', 'AlcoholDrinking o'))

table(df$AlcoholDrinking , df$Sex)
table(df$AlcoholDrinking , df$Stroke)
table(df$AlcoholDrinking , df$BMI)
exp(0.39)

str(df2$Ageca)
anova.glm = anova(glm.df6, test= 'Chisq')
