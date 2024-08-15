
library(readr)
library(dplyr)
library(quickpsy)
library(ggplot2)

# Set directories
DIR <- setwd("C:/Users/hinne/Desktop/BA Voice/EEG Data analysis EEG/Voice analysis/Behavioural Data CSV")
behavior_results_DIR <- file.path(DIR)

# Define subjects
subjs <- c('sub_08', 'sub_09', 'sub_10', 'sub_11', 'sub_07', 'sub_12',
           'sub_13', 'sub_14', 'sub_15', 'sub_18', 'sub_19', 'sub_20', 
           'sub_21', 'sub_22', 'sub_23', 'sub_24', 'sub_25', 'sub_26', 
           'sub_27', 'sub_28')

# Read and combine data
appended_data <- lapply(subjs, function(subj) {
  data <- read_csv(file.path(behavior_results_DIR, paste0(subj, '_summarized-behavior.csv')))
  data$subj <- subj
  return(data)
})
df_behav <- bind_rows(appended_data)

#responses_num <- as.numeric(df_behav$Voice_responses)

names(df_behav)[names(df_behav) == "Voice responses"] <- "Voice_responses"
names(df_behav)[names(df_behav) == "Morph ratio"] <- "Morph_ratio"


# Fit the Psychometric Function and get the PSE
results_PSE <- df_behav %>%
  group_by(subj) %>%
  do({
    model <- quickpsy(., x = Morph_ratio, k = Voice_responses, n = N, 
                      fun = logistic_fun)
    data.frame(PSE = model$par$par[1])
  })

# Plot the psychometric function over subjects
data <- df_behav %>%
  group_by(subj) %>%
  summarise(Morph_ratio = mean(Morph_ratio), Voice_responses = mean(Voice_responses), N = mean(N))

#na_check <- data %>%
  #filter(is.na(Morph_ratio) | is.na(Voice_responses) | is.na(N))

#print(na_check)

# If there are NA values, you need to handle them (e.g., remove or impute)
#data_clean <- na.omit(data)

# Fit a global model for plotting
global_model <- quickpsy(df_behav, x = Morph_ratio, k = Voice_responses, n = N, fun = logistic_fun)

# Plotting the psychometric function
plotcurves(global_model, color = "red") +
  labs(x = "%Voice Responses", y = "Morph Ratio")
  #geom_line(data = global_model, linewidth = 10, color = "red")



# Extract the data for the fitted psychometric curves
fitted_curves <- global_model$curves
 global_model$averages

# Plotting with ggplot2
ggplot(fitted_curves, aes(x = x, y = y)) +
  geom_line(color = "#3CC3C8", size = 1.5) +  # Customize line color and width
  labs(x = "Morph Ratio", y = "Voice Responses") +
  theme_minimal() +
  theme(axis.title.x = element_text(size = 18),
        axis.title.y = element_text(size = 18),
        axis.text = element_text(size = 12))


#Boxplot
df_behav$Morph_ratio <- as.factor(df_behav$Morph_ratio)

names(df_behav)[names(df_behav) == "%Voice"] <- "Perc_Voice"

# Create boxplots of Voice_responses by Morph_ratio
ggplot(df_behav, aes(x = Morph_ratio, y = Perc_Voice)) +
  geom_boxplot(fill = "#3CC3C8") +
  labs(
       x = "Morph Ratio",
       y = "Voice Responses") +
  scale_y_continuous() +  # Format y-axis as percentages
  theme_minimal() +
  theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1),
        axis.text.y = element_text(size = 12),
        axis.title.x = element_text(size = 18),
        axis.title.y = element_text(size = 18))
