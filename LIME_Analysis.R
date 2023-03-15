# LIME Analysis of Fatal Terrorism

# We want to know what factors that drive terrorist groups to conduct terrorist attacks.
# We will use LIME algorithym. LIME stands for Local Interpretable Model-Agnostic Explanations. Local interpretations help us understand model predictions for a 
# single row of data or a group of similar rows.

# Clear Workspace ---------------------------------------------------------

rm(list = ls()) # clear global environment to remove all data sets, functions and so on.

# Libraries ---------------------------------------------------------------

# install vip from github repo: devtools::install_github("koalaverse/vip")
library(lime)       # ML local interpretation
library(vip)        # ML global interpretation
library(pdp)        # ML global interpretation
library(ggplot2)    # visualization pkg leveraged by above packages
library(caret)      # ML model building
library(h2o)        # ML model building
library(xgboost)    # ML model building
library(tidyverse)  # Use tibble, dplyr
library(gridExtra)  # Plot multiple lime plots on one graph
library(purrr)      # for automating data
library(Hmisc)      # for data exploration
library(data.table) # for set.names function
library(modelr)     # for add_predictions function

# initialize h2o
h2o.init()

# Set Working Directory ---------------------------------------------------

setwd("C:/GTD")

# Load Data ---------------------------------------------------------------

GTD <- read.csv("globalterrorismdb_0919dist.csv")

# Data Wrangling ----------------------------------------------------------

################################
# Filter Doubt Terrorism == No #
################################

# This removes all the non - terrorism violence as such incidents are coded as 1.
# A code of 1 means it is doubted such violence is terrorism, while the zero code indicates an attack is terrorist. Therefore, we select only the rows with a zero 
# value.

GTDDT <- GTD %>% dplyr::filter(doubtterr == 0)
str(GTDDT)
dim(GTDDT)

###########################
# Filter Specificity == 1 #
###########################

# This removes all rows where the geographic coordinates have not been verified
# This is important because province and city variables are used in the modeling, so it is necessary to know exactly where each attack occurred.

GTDDTS <- GTDDT %>% dplyr::filter(specificity == 1)

#########################
# Change Variable Names #
#########################

setnames(GTDDTS, old = c("iyear", "imonth", "iday", "extended", "country_txt",  "region_txt", "provstate", "city", "success", "multiple", "suicide", 
                         "attacktype1_txt", "gname", "targtype1_txt", "weaptype1_txt", "nkill", "nwound"), new = c("Year", "Month", "Day", "Extended", "Country",  
                                                                                                                   "Region", "Province", "City", "Success", 
                                                                                                                   "Multiple", "Suicide", "Attack", "Group", 
                                                                                                                   "Target", "Weapon", "Dead", "Wounded"))

####################
# Remove NA Values #
####################

# Here, we will convert NA values into zeros, so we don't lose info from the rest of any data row

colSums(is.na(GTDDTS))
GTDDTS$Dead[is.na(GTDDTS$Dead)] <- 0

###########################
# Select Specific Columns #
###########################

# We are only interested in the year, country, attack type, province, city, terrorist group, attack target and a variable with counts of terrorist attack fatalities.

GTDDTS2 <- dplyr::select(GTDDTS, "Year", "Country", "Attack", "Weapon", "Province", "City", "Group", "Target", "Dead")

#########################
# Country Attack Counts #
#########################

# We create the below Country_Count object below, which lists the attack counts per country in descending order in a table.
# We see that Iraq, Pakistan, India, Afghanistan and Colombia comprise the five countries with the highest terrorist attack counts. Chile, which rarely features 
# in news about terrorism is listed as the 15th country with most attacks, so let's select that. Additionally, it is still numerous enough to provide a good sample.

Country_Count <- map_dfr(
  .x = quos(Country),
  .f = function(x, data = GTDDTS2) {
    GTDDTS2 %>% 
      count({{ x }}, sort = T) %>% 
      mutate(variable = names(.)[1]) %>% 
      rename(category = 1) %>% 
      select(variable, category, n)
    
  }
)

Country_Count

#####################
# Filter Chile Data #
#####################

Chile_Data <- GTDDTS2 %>% dplyr::filter(Country == "Chile")
glimpse(Chile_Data)

##############################################
# Convert Lethal Variable into binary format #
##############################################

# Here we create a new Lethal terrorist attack column. This variable is populated by zero and one values. If values in the old Dead column are zero, then they 
# remain zero in the Lethal column, For all values other than zero in the Dead column, these have been transformed into a value of one in the Lethal attack column.

Chile_Data_final <- Chile_Data %>% dplyr::mutate(Lethal = if_else(Dead == 0, "0", "1"))

###########################
# Remove Unneeded Columns #
###########################

# We remove the year, country and weapon columns. This is required, as we are not interested in annual analysis, we don't need a country column as we are only 
# dealing with Chile data and weapon is correlated with the attack variable. We also don't need the counts of dead in the Dead variable, as we are now interested in 
# predicting fatality yes or no. 

Chile_Data_final <- Chile_Data_final %>% dplyr::select(-c(Year, Country, Weapon, Dead))
names(Chile_Data_final)

############################
# Convert columns to factors
############################

Chile_Data_final$Lethal <- as.factor(Chile_Data_final$Lethal)

###############
# Saving Data #
###############

# Save to a file

saveRDS(Chile_Data_final, "Chile_Data_final.rds")

# Restore it 

Chile_Data_final <- readRDS("Chile_Data_final.rds")

######################
# Remove Cardinality #
######################

# Factor Variables have been amended, whereby all variable levels with a count of less than 5% of the total count of the variable have been removed. This is done to 
# enhance data visualisation of the plots and interpretability of the results.

factor_columns <- c('Province','City', 'Group', 'Target') # create vector of named columns

Chile_RC <- Chile_Data_final %>%  mutate(across(factor_columns, fct_lump_prop, prop = 0.05, other_level = 'Other'))

################################
# Recode 'Other' Factor Levels #
################################

Chile_RC$Province <- recode_factor(Chile_RC$Province, Other = "OtherProvince") 
Chile_RC$City <- recode_factor(Chile_RC$City, Other = "OtherCity") 
Chile_RC$Group <- recode_factor(Chile_RC$Group, Other = "OtherGroup") 
Chile_RC$Target <- recode_factor(Chile_RC$Target, Other = "OtherTarget") 

glimpse(Chile_RC)

###############################
# Recode Lethal Factor Levels #
###############################

Chile_RC <- Chile_RC %>%
  mutate(Lethal = fct_recode(Lethal, 
                                  "No" = "0", 
                                  "Yes" = "1"))

# Save to a file#

saveRDS(Chile_RC, "Chile_RC.rds")

# Restore it #

Chile_RC <- readRDS("Chile_RC.rds")

# Data Exploration --------------------------------------------------------

# This section provides code which documents various statistical summaries of the data. This allows the user to understand the various dimensions and 
# characteristics of the data.

Hmisc::describe(Chile_RC)
skimr::skim(Chile_RC) # provides a data summary
DataExplorer::introduce(Chile_RC) # provides a data summary
DataExplorer::plot_intro(Chile_RC) # bar plot of percentage of discrete and continous columns, missing columns, complete rows and missing observations
DataExplorer::plot_missing(Chile_RC) # indicates percentage of missing rows per column
DataExplorer::plot_bar(Chile_RC) # distribution of categorical variables
DataExplorer::plot_histogram(Chile_RC) # distribution of numerical variables
DataExplorer::plot_qq(Chile_RC)

# Data Splitting ----------------------------------------------------------

# It is important to use stratified sampling. This technique consists of forcing the distribution of the target variable(s) among the different splits to be the 
# same. Otherwise, there is an imbalance in the data, which may interfere with how the algorithm generates predictions, as it will favour the class of the target 
# variable with the highest count. Stratified sampling can be carried out with the createDataPartition function in the caret package. We create a slit of 75% for
# training data and 25% for test data. We specify we don't want a list using the list = False() parameter

set.seed(123)
Chile_DP <- createDataPartition(y = Chile_RC$Lethal, p = .75, list = F)
Chile_train <- Chile_RC[Chile_DP,]
Chile_test <- Chile_RC[-Chile_DP,]
dim(Chile_test)

# Modelling ---------------------------------------------------------------

# Create Random Forest model via caret

fit.rf_Chile <- train(
  Lethal ~ ., 
  data = Chile_train, 
  method = 'rf',
  trControl = trainControl(method = "cv", number = 10))

# Predictions -------------------------------------------------------------

Chile_rf_pred <- predict(fit.rf_Chile, Chile_test)
confusionMatrix(Chile_rf_pred, as.factor(Chile_test$Lethal))

# Global Interpretations --------------------------------------------------

# A common ways of obtaining global interpretation is through:

# variable importance measures
# Variable importance quantifies the global contribution of each input variable to the predictions of a machine learning model. Variable importance measures rarely 
# give insight into the average direction that a variable affects a response function. They simply state the magnitude of a variable’s relationship with the 
# response as compared to other variables used in the model. 

# Variable Importance Measures #

# The below plot indicates that police target, assassination attack and bomb attack are the three most important variables in terms of predicting Lethal terrorist 
# attack

vip(fit.rf_Chile) + ggtitle("Ranger Random Forest Model") + theme_classic() + theme(plot.title = element_text(hjust = 0.5)) 

# Local Interpretations ---------------------------------------------------

explainer_caret_Chile <- lime(Chile_train %>% select(-Lethal), fit.rf_Chile)

# Let us pick the first 6 rows.
# We have selected the five best variables to use.
# We have selected the label of lethal attack as yes.

set.seed(123)
explanation_reason_ridge_caret_Chile <- lime::explain(
  x = Chile_test[c(1:6),] %>% select(-Lethal), 
  explainer = explainer_caret_Chile, 
  n_permutations = 5000,
  dist_fun = "gower",
  kernel_width = .75,
  n_features = 5, 
  feature_select = "highest_weights", # Fit ridge regression and select n_features with highest absolute weight
  n_labels = 1) # Indicates the probability of an attack being lethal
tibble::glimpse(explanation_reason_ridge_caret_Chile)
View(explanation_reason_ridge_caret_Chile)
Chile_test$Lethal

explanation_reason_ridge_caret_Chile_DF <- as.data.frame(explanation_reason_ridge_caret_Chile)
glimpse(explanation_reason_ridge_caret_Chile_DF)

# For some parameters it is possible to adjust in explanation function:

# x = The object you want to explain
# labels = What specific labels of the target variables you want to explain
# explainer = the explainer object from lime function
# n_features = number of features used to explain the data
# n_permutations = number of permutations for each observation for explanation. The default is 5000 permutations
# dist_fun = distance function used to calculate the distance to the permutation. The default is Gower’s distance but can also use euclidean, manhattan, or any 
# other distance function allowed by ?dist()
# kernel_width = An exponential kernel of a user defined width (defaults to 0.75 times the square root of the number of features) used to convert the distance
# measure to a similarity value

plot_features(explanation_reason_ridge_caret_Chile)

# The text label '1' shows what value of target variable is being explained, which in this case, we are interested in explaining the predictions where a terrorist 
# attack is not lethal. The probability shows the probability of the observation belong to the label no. Inside each plot, there are several bar charts. The y-axis 
# indicates the key features while the x-axis show the relative strength of each features. The positive value (blue color) indicates that the feature supports or 
# increase the value of the prediction, while the negative value (red color) has a negative effect or decrease the prediction value.
# ALl the predicted probabilities of non lethal attack are high, which is just as well, as when we compare these predictions with the actual values in the test data, 
# for these five rows, every row is not lethal attack:

# We can compare the results with the actual test data values:
  
head(Chile_test$Lethal)  

# Surprisingly, in the three rows where bomb attack is one of the five key variables, it signficantly supports the prediction of non lethal attack. Assassination 
# attack strongly contradicts the prediction of non lethal attack. In all cases, Santiago city supports the prediction of lethal attack, as does all groups apart 
# from the Movement of the Revolutionary Left. All target variables positively supported the prediction of lethal attack.
