# Load necessary libraries
library(dplyr)
library(ggplot2)
library(multcomp) # For Tukey's HSD 

# read data
data <- read.csv("C:/Users/hinne/Desktop/BA Voice/EEG Data analysis EEG/Voice analysis/R analysis/Data/EEG_data_Cz.csv", header=T, na.strings = "NaN")


print(class(data))

# Convert Condition to a factor
data$condition <- as.factor(data$condition)

# Perform ANOVA
anova_result <- aov(amp ~ condition, data = data)

# Print the ANOVA summary
summary(anova_result)


# Perform Tukey's HSD test if ANOVA is significant
tukey_result <- TukeyHSD(anova_result)

# Print Tukey's HSD results
print(tukey_result)
