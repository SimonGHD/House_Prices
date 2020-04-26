############################################################################################################
### Predict the house price using XAI in an environment
############################################################################################################

# This Skript aims to provide a workflow with a closed environment and  XAI on Predicting  Task using models, such as XGBOOST
# The Preprocessing and first models was obtained from :https://www.kaggle.com/jiashenliu/updated-xgboost-with-parameter-tuning


if (!requireNamespace("remotes"))
  install.packages("remotes")

remotes::install_github("rstudio/renv")

setwd("C:/Users/simon/R-Projekte/HP_XAI/House_Prices")
renv::init()

# Load Packages
install.packages("MASS", type = "win.binary")
install.packages("Metrics", type = "win.binary")
install.packages("corrplot", type = "win.binary")
install.packages("randomForest", type = "win.binary")
install.packages("lars", type = "win.binary")
install.packages("ggplot2", type = "win.binary")
install.packages("xgboost", type = "win.binary")
install.packages("Matrix", type = "win.binary")
install.packages("methods", type = "win.binary")
install.packages("caret", type = "win.binary")
install.packages("devtools", type = "win.binary")
install.packages("backports", type = "win.binary")
install.packages("rlang", type = "win.binary")
install.packages("DALEX", type = "win.binary")
install_dependencies()
install.packages("iBreakDown", type = "win.binary")

require("MASS") 
require("Metrics")
require("corrplot")
require("randomForest")
require("lars")
require("ggplot2")
require("xgboost")
require("Matrix")
require("methods")
require("caret")
require("iBreakDown")

getwd()
# Read Data
Training <- read.csv("train.csv")
Test <- read.csv("test.csv")
# Test whether data is successfully loaded
names(Training)


# After that, the whole procedure has begun. I divide the whole process into four steps:
#   
# * Data Cleansing
# * Descriptive Analysis
# * Model Selection
# * XAI

############################################################################################################
### Data Cleansing
############################################################################################################

Num_NA<-sapply(Training,function(y)length(which(is.na(y)==T)))
NA_Count<- data.frame(Item=colnames(Training),Count=Num_NA)
NA_Count

# Among 1460 variables, "Alley",  "PoolQC", "Fence" and "MiscFeature" have amazingly high number of missing value. Therefore, I 
# have decided to remove those variables. After that, the number of effective variables has shrunken to 75 (excluding id). 

Training<- Training[,-c(7,73,74,75)]

# Numeric Variables
Num<-sapply(Training,is.numeric)
Num<-Training[,Num]

for(i in 1:77){
  if(is.factor(Training[,i])){
    Training[,i]<-as.integer(Training[,i])
  }
}

# Test
Training$Street[1:50]
# Replacing Missing values with 0
Training[is.na(Training)]<-0
Num[is.na(Num)]<-0

############################################################################################################
### Descriptive Analysis
############################################################################################################

# Exploring dataset could be diffcult when the quantity of variables is quite huge. Therefore, I mainly focused on the exploration of numeric
# variables in this report. The descriptive analysis of dummy variables are mostly finished by drawing box plots. Some dummy variables, like "Street",
# are appeared to be ineffective due to the extreme box plot. The numeric variables are sorted out before turning dummy variables into numeric form.

# We first draw a corrplot of numeric variables. Those with strong correlation with sale price are examined.
#```{r,message=FALSE, warning=FALSE}
correlations<- cor(Num[,-1],use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")
#
# "OverallQual","TotalBsmtSF","GarageCars" and "GarageArea" have relative strong correlation with each other. Therefore, as an example, we plot the correlation
# among those four variables and SalePrice.
pairs(~SalePrice+OverallQual+TotalBsmtSF+GarageCars+GarageArea,data=Training,
      main="Scatterplot Matrix")


############################################################################################################
### Model Selection
############################################################################################################

# Before implementing models, one should first split the training set of data into 2 parts: a training set within the training set and a test set that can be used for evaluation.
# Personally I prefer to split it with the ratio of 6:4, ***But if someone can tell me what spliting ratio is proved to be scienticfic I will be really grateful***
##   
#  ```{r,message=FALSE, warning=FALSE}
# Split the data into Training and Test Set # Ratio: 6:4 ###
Training_Inner<- Training[1:floor(length(Training[,1])*0.7),]
Test_Inner<- Training[(length(Training_Inner[,1])+1):length(Training),]

# I will fit three regression models to the training set and choose the most suitable one by checking RMSE value.

############################################################################################################
#### Model 1: Linear Regression
############################################################################################################

# The first and simplest but useful model is linear regression model. As the first step, I put all variables into the model.


# R Square is not bad, but many variables do not pass the Hypothesis Testing, so the model is not perfect. Potential overfitting will occur if someone insist on using it. Therefore,
# the variable selection process should be involved in model construction. I prefer to use Step AIC method.

# Several variables still should not be involved in model. By checking the result of Hypothesis Test, I mannually build the final linear regression model.

#```{r,message=FALSE,warning=FALSE}
reg1_Modified_2<-lm(formula = SalePrice ~ MSSubClass + LotArea + 
                      Condition2 + OverallQual + OverallCond + 
                      YearBuilt  + RoofMatl +  ExterQual + 
                      BsmtQual + BsmtCond + BsmtFinSF1 + BsmtFinSF2 + 
                      BsmtUnfSF + X1stFlrSF + X2ndFlrSF + BedroomAbvGr + KitchenAbvGr + 
                      KitchenQual + TotRmsAbvGrd + Functional + Fireplaces + FireplaceQu + 
                      GarageYrBlt + GarageCars +  SaleCondition, 
                    data = Training_Inner)
summary(reg1_Modified_2)

# The R Square is not bad, and all variables pass the Hypothesis Test. The diagonsis of residuals is also not bad. The diagnosis can be viewed below.
#```{r,message=FALSE,warning=FALSE}
layout(matrix(c(1,2,3,4), 2, 2, byrow = TRUE))
plot(reg1_Modified_2)
par(mfrow=c(1,1))


# We check the performance of linear regression model with RMSE value.

Prediction_1<- predict(reg1_Modified_2, newdata= Test_Inner)
rmse(log(Test_Inner$SalePrice),log(Prediction_1))
#```

############################################################################################################
#### Model 2: LASSO Regression
############################################################################################################

# For the avoidance of multicollinearity, implementing LASSO regression is not a bad idea. Transferring the variables into the form of matrix, we can automate
# the selection of variables by implementing "lars" method in Lars package.

Independent_variable<- as.matrix(Training_Inner[,1:76])
Dependent_Variable<- as.matrix(Training_Inner[,77])
laa<- lars(Independent_variable,Dependent_Variable,type = "lasso")
plot(laa)


# The plot is messy as the quantity of variables is intimidating. Despite that, we can still use R to find out the model with least multicollinearity. The selection 
# procedure is based on the value of Marrows cp, an important indicator of multicollinearity. The prediction can be done by the script-chosen best step and RMSE can be used
# to assess the model.

#```{r,message=FALSE,warning=FALSE}
best_step<- laa$df[which.min(laa$Cp)]
Prediction_2<- predict.lars(laa,newx =as.matrix(Test_Inner[,1:76]), s=best_step, type= "fit")
rmse(log(Test_Inner$SalePrice),log(Prediction_2$fit))
#```

############################################################################################################
#### Model 3: Random Forest
############################################################################################################

# The other model I chose to fit in the training set is Random Forest model. The model, prediction and RMSE calculation can be found below:

#```{r,message=FALSE, warning=FALSE}
for_1<- randomForest(SalePrice~.,data= Training_Inner)
Prediction_3 <- predict(for_1, newdata= Test_Inner)
rmse(log(Test_Inner$SalePrice),log(Prediction_3))
#```

# Obviously, Random Forest may produce the best result within the training set so far. 

############################################################################################################
#### Model 4: XGBoost 
############################################################################################################

# This amazing package really impressed me! And I have enthusiam to explore it. The first step of XGBoost is to transform the dataset into Sparse matrix.
#

train<- as.matrix(Training_Inner, rownames.force=NA)
test<- as.matrix(Test_Inner, rownames.force=NA)
train <- as(train, "sparseMatrix")
test <- as(test, "sparseMatrix")
# Never forget to exclude objective variable in "data option"
train_Data <- xgb.DMatrix(data = train[,2:76], label = train[,"SalePrice"])

# Then I tune the parameters of xgboost model by building a 20-iteration for-loop. **Not sure whether this method is reliable but really time-consuming**
# **Updated** Thanks for the advices from my fellow Kaggle friend! Now I understand how to use "Caret" to perform grid search for the parameters. 
#```{r,message=FALSE,warning=FALSE}
# Tuning the parameters #
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3)

xgb.grid <- expand.grid(nrounds = 25,
                        max_depth = seq(6,10),
                        eta = c(0.2,0.3, 1),
                        gamma = c(0.0, 0.2, 1),
                        colsample_bytree = c(0.5,0.8, 1),
                        min_child_weight=seq(1,10),
                        subsample = 1
)

xgb_tune <-train(SalePrice ~.,
                 data=Training_Inner,
                 method="xgbTree",
                 metric = "RMSE",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid
)

print(xgb.grid)
print(xgb_tune)

tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) +
    theme_bw()
}

tuneplot(xgb_tune)

# Then, the parameter can be selected by the random process. Since the process is relatively boring, I just skip it in RMarkdown file and use the optimal parameters 
# I got in my local R script for the prediction and evaluation. ** Can I ask some more efficient and intelligent method of parameter tuning from smart Kagglers?
# Looking forward to your advice!!**

# The model should be tested before making actual prediction.


test_data <- xgb.DMatrix(data = test[,2:76])

prediction <- predict(xgb_tune, Test_Inner)
rmse(log(Test_Inner$SalePrice),log(prediction))



############################################################################################################
# **** XAI-PArt****  
############################################################################################################

# explainer
# Dalex primary Operations
# 1. Any supervised regression or binary classification model with defined input (X) and output (Y) where the output can be customized to a defined format can be used.
# 2. The machine learning model is converted to an "explainer" object via DALEX::explain(), which is just a list that contains the training data and meta data on the machine learning model.
# 3. The explainer object can be passed onto multiple functions that explain different components of the given model.
library("DALEX")

explainer_reg <- explain(reg1_Modified_2, 
                          data = Training, 
                          y = Training$SalePrice, 
                          label = "lreg")


explainer_rf <- explain(for_1, 
                        data = Training, 
                        y = Training$SalePrice, 
                        label = "random_forest")


explainer_xgb <- explain(xgb_tune, 
                         data = Training, 
                         y = Training$SalePrice, 
                         label = "xgboost")

############################################################################################################
# *** Residual Analysis
############################################################################################################

#However, a single accuracy metric can be a poor indicator of performance. 
#Assessing residuals of predicted versus actuals can allow you to identify where models deviate in their predictive accuracy. 

#And looking at the boxplots you can see that the GBM model also had the lowest median absolute residual value. 
#Thus, although the GBM model had the lowest AUC score, it actually performs best when considering the median absoluate residuals. 
#However, you can also see a higher number of residuals in the tail of the GBM residual distribution (left plot) suggesting that there may be a higher number of large residuals compared to the GLM model. 
#This helps to illustrate how your residuals behave similarly and differently across models.

resids_reg <- model_performance(explainer_reg) 

resids_lars <- model_performance(explainer_lars) 
#--> not working
resids_rf  <- model_performance(explainer_rf)

resids_xgb <- model_performance(explainer_xgb)



p1 <- plot(resids_reg, resids_rf, resids_xgb)
p2 <- plot(resids_reg, resids_rf, resids_xgb, geom = "boxplot")

gridExtra::grid.arrange(p1, p2, nrow = 1)


############################################################################################################
# *** Variable IMportance 
############################################################################################################

#DALEX uses a model agnostic variable importance measure computed via permutation. 
#This approach follows the following steps:
  
# For any given loss function do
# 1: compute loss function for full model (denote _full_model_)
# 2: randomize response variable, apply given ML, and compute loss function (denote _baseline_)
# 3: for variable j
# | randomize values
# | apply given ML model
# | compute & record loss function
# end

# Plot interpretation:
# Left edge of x-axis is the loss function for the _full_model_. 
# The default loss function is squared error but any custom loss function can be supplied.
# The first item listed in each plot is _baseline_. 
# This value represents the loss function when our response values are randomized and should be a good indication of the worst-possible loss function value when there is no predictive signal in the data.
# The length of the remaining variables represent the variable importance. 
# The larger the line segment, the larger the loss when that variable is randomized.



#vip_lars <- variable_importance(explainer_lars, n_sample = 1000, loss_function = loss_root_mean_square) 

vip_lreg  <- variable_importance(explainer_reg, n_sample = 1400, loss_function = loss_root_mean_square)
vip_rf  <- variable_importance(explainer_rf, n_sample = 1400, loss_function = loss_root_mean_square)
vip_xgb <- variable_importance(explainer_xgb, n_sample =1400, loss_function = loss_root_mean_square)

plot(vip_lreg, vip_rf, vip_xgb, max_vars = 10)

############################################################################################################
# *** Predictor-response relationship
############################################################################################################

# understand how the relationship between these influential variables and the predicted response differ between the models
# indicate if each model is responding to the predictor signal similarly or if one or more models respond differently

# The below partial dependence plot illustrates that the GBM and random forest models are using the Age signal in a similar non-linear manner; 
# however, the GLM model is not able to capture this same non-linear relationship. 
# So although the GLM model may perform better (re: AUC score), it may be using features in biased or misleading ways.
# alt: pdp_xgb <- variable_response(explainer_xgb, variable =  "OverallQual", type = "pdp")

pdp_lreg   <- model_profile(explainer_reg,  variable =  "OverallQual")

pdp_rf   <- model_profile(explainer_rf,  variable =  "OverallQual")
pdp_xgb <- model_profile(explainer_xgb, variable =  "OverallQual")

plot(pdp_lreg$agr_profiles, pdp_rf$agr_profiles, pdp_xgb$agr_profiles) +
  ggtitle("Contrastive Partial dependence profiles") 
#--> lreg not able to capture non-linearity

bd_rf <- variable_attribution(explainer_rf,
                              new_observation = henry,
                              type = "break_down")

############################################################################################################
# *** Local-dependence and Accumulated Local Profiles
############################################################################################################

pd_lreg <- variable_profile(explainer_reg, 
                          type = "partial",
                          variables = c("OverallQual", "YearBuilt"))
pd_rf <- variable_profile(explainer_rf, 
                          type = "partial",
                          variables = c("OverallQual", "YearBuilt"))
pd_xgb <- variable_profile(explainer_xgb, 
                          type = "partial",
                          variables = c("OverallQual", "YearBuilt"))

plot(pd_lreg) +
  ggtitle("Partial dependence for OverallQual and YearBuilt") 

plot(pd_rf) +
  ggtitle("Partial dependence for OverallQual and YearBuilt") 

plot(pd_xgb) +
  ggtitle("Partial dependence for OverallQual and YearBuilt") 


############################################################################################################
############################################################################################################
# Local Level
############################################################################################################
############################################################################################################

############################################################################################################
# *** Ceteris-paribus Profiles
############################################################################################################

# In this chapter we focus on methods that analyse an effect of selected variables on model response.
#  examine the influence of each explanatory variable, assuming that effects of all other variables are unchanged. 
# The main goal is to understand how changes in a single explanatory variable affects model predictions
# We are interested in the change of the model prediction induced by each of the variables. 
# Toward this end, we may want to explore the curvature of the response surface around a single point with age equal to 47 and class equal to "1st," indicated in the plot.

# The difference between these two methods lies in the fact that LIME approximates the model of interest locally with a simpler glass-box model.

predict(explainer_rf, Training[1,])


cp_rf <- individual_profile(explainer_rf, 
                                    new_observation = Training[1,])


plot(cp_rf, variables = c("OverallQual", "LotFrontage")) +
  ggtitle("Ceteris Paribus Profile", 
          "For the random forest model and the dataset")



############################################################################################################
# *** Ceteris-paribus Profiles
############################################################################################################

predict(explainer_rf, Training[1,])

shap_rf_1 <- variable_attribution(explainer_rf, 
                                   Training[1,], 
                                   type = "shap",
                                   B = 25)
plot(shap_rf_1) 

shap_xgb_1 <- variable_attribution(explainer_xgb, 
                                    Training[1,], 
                                   type = "shap",
                                   B = 25)
plot(shap_xgb_1) 


############################################################################################################
# Environment Management
############################################################################################################
renv::snapshot()


  