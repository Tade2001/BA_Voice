# LMER ANALYSIS EEG VOICE 
library(BayesFactor)
library(Matrix)
library(plotrix)
library(lme4)
library(lmerTest)
library(emmeans)
library(afex)
library(car)
library(sjPlot)
library(brms)
library(plotly) 
library(performance)
library(ggeffects)  
require(gridExtra)
  

# read data
data <- read.csv("C:/Users/hinne/Desktop/BA Voice/EEG Data analysis EEG/Voice analysis/R analysis/Data/EEG_data_FC5.csv", header=T, na.strings = "NaN")

data$subj <- as.factor(data$subj)

# Scale the amplitude
data$ampz = scale(data$amp)
# Scale the behaviour
data$X.Voicez = scale(data$X.Voice)


#####################################################

# Check distribution of response variable
hist(data$X.Voicez)
# Check distribution of explanatory variable 
hist(data$ampz) #amplitude seems to be more or less normally distributed

# Clustered data?!
(split_plot <- ggplot(aes(ampz, X.Voicez), data = data) + 
    geom_point() + 
    facet_wrap(~ subj) + # create a facet for each mountain range
    xlab("X.Voicez") + 
    ylab("ampz"))


# MIXED EFFECTS MODEL 1
mixed.lmer <- lmer(X.Voice ~ amp + condition + (1|subj), data=data)
summary(mixed.lmer)



# Check the assumptions 
plot(mixed.lmer)
qqnorm(resid(mixed.lmer))
qqline(resid(mixed.lmer)) 

# Plotting the model predictions
# Extract the prediction data frame
pred.mm <- ggpredict(mixed.lmer, terms = c("amp"))  

# Plot the predictions-Overall
fig1<-(ggplot(pred.mm) + 
         geom_line(aes(x = x, y = predicted)) +      
         geom_ribbon(aes(x = x, ymin = predicted - std.error, ymax = predicted + std.error), 
                     fill = "lightgrey", alpha = 0.5) +  # error band
         geom_point(data = data,                      # adding the raw data (scaled values)
                    aes(x = amp, y = X.Voice, colour = subj)) + 
         labs(x = "Amplitude", y = "Voice response", 
              title = "FC6") + 
         theme_minimal() +
         theme(
           plot.title = element_text(size = 22),      
           axis.title.x = element_text(size = 18),     
           axis.title.y = element_text(size = 18),     
           axis.text = element_text(size = 14),
           legend.title = element_text(size = 16),     
           legend.text = element_text(size = 12))
)
fig1 

# Results table
tab_model(mixed.lmer) 

# BASELINE MODEL
mixed.lmer2 <- lmer(X.Voice ~  condition  + (1|subj), data=data)
summary(mixed.lmer2)
plot(mixed.lmer2)
qqnorm(resid(mixed.lmer2))
qqline(resid(mixed.lmer2)) # Residuals are not normally distributed!


# Comparison to reduced model: Drop fixed effect for reduced model 
anova(mixed.lmer, mixed.lmer2) 

################################################################################


(split_plot <- ggplot(aes(ampz, X.Voicez), data = data) + 
   geom_point() + 
   facet_wrap(~ subj) + # create a facet for each mountain range
   xlab("condition") + 
   ylab("ampz"))


# MIXED EFFECTS MODEL 1
mixed.lmer <- lmer(X.Voicez ~ ampz+ condition + (1|subj), data=data)
summary(mixed.lmer)


# Check the assumptions 
plot(mixed.lmer)
qqnorm(resid(mixed.lmer))
qqline(resid(mixed.lmer)) 

# Plotting the model predictions
# Extract the prediction data frame
pred.mm <- ggpredict(mixed.lmer, terms = c("ampz"))  

# Plot the predictions-Overall
fig1<-(ggplot(pred.mm) + 
         geom_line(aes(x = x, y = predicted)) +      
         geom_ribbon(aes(x = x, ymin = predicted - std.error, ymax = predicted + std.error), 
                     fill = "lightgrey", alpha = 0.5) +  # error band
         geom_point(data = data,                      # adding the raw data (scaled values)
                    aes(x = ampz, y = X.Voicez, colour = subj)) + 
         labs(x = "condition", y = "ampz", 
              title = "") + 
         theme_minimal()
)
fig1

fig3<-(ggplot(pred.mm) +
         geom_line(aes(x = x, y = predicted)) +      
         geom_ribbon(aes(x = x, ymin = predicted - std.error, ymax = predicted + std.error), 
                     fill = "lightgrey", alpha = 0.5) +  # error band
        geom_point(data = data,                      # adding the raw data (scaled values)
           aes(x = condition, y = ampz, colour = subj)) + 
        labs(x = "condition", y = "ampz", 
         title = "") + 
         scale_x_continuous(limits = c(min_value=0, max_value=1),  # Set min and max limits
                            #breaks = seq(from, to, by = interval),  # Set specific breaks
                            #labels = c("Label1", "Label2", ...) 
                            )+
        theme_minimal())
fig3 

# Results table
tab_model(mixed.lmer) 

# BASELINE MODEL
mixed.lmer2 <- lmer(X.Voicez ~  condition  + (1|subj), data=data)
summary(mixed.lmer2)
plot(mixed.lmer2)
qqnorm(resid(mixed.lmer2))
qqline(resid(mixed.lmer2)) # Residuals are not normally distributed!


# Comparison to reduced model: Drop fixed effect for reduced model 
anova(mixed.lmer, mixed.lmer2) 




