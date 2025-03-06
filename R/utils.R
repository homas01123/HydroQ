# Import the in situ data and perform Explanatory Data Analysis (EDA) ----

# library(tidyverse)
# long_data = read.csv("./data/do_secondsource_long.csv", header = T)
# 
# long_data = long_data[(long_data$CharacteristicName == "Chlorophyll a" & 
#                         long_data$ResultSampleFraction == "Total") | 
#                         (long_data$CharacteristicName == "Ammonia" ) |
#                         (long_data$CharacteristicName == "pH") , (1:2)]
# 
# # Step 1: Create a row index for each occurrence of a characteristic
# long_data <- long_data %>%
#   group_by(CharacteristicName) %>%
#   mutate(row_id = row_number()) %>%
#   ungroup()
# 
# # Step 2: Pivot to wide format using row_id to align values properly
# wide_data <- long_data %>%
#   pivot_wider(names_from = CharacteristicName, values_from = ResultValue) %>%
#   select(-row_id)  # Remove row index if not needed
# 
# # View output
# print(wide_data)
# 
# wide_data$`Chlorophyll a` = wide_data$`Chlorophyll a`*0.001
# 
# chl_do_inland = read.csv("./data/ChlaData.csv", header = T)
# chl_do_inland = chl_do_inland[,c(seq(1,9,1),13,19)]
# chl_do_inland  = chl_do_inland[complete.cases(chl_do_inland),]
# write.csv(chl_do_inland, "./data/chl_do_inland.csv", quote = F, row.names = F)


do_merged = read.csv("./data/insitu_wq_data.csv", header = T)
#cor(do_merged$water_temp_celcius, do_merged$doxy_mg_L, use = "pairwise")

# Load required libraries
library(tidyverse)
library(ggplot2)
library(skimr)
library(GGally)  # For pairwise plots
library(corrplot)  # For correlation matrix
library(explore)  # Automated EDA

# Load your dataset (assuming it's named df)
df <- do_merged  # Based on your screenshot

# ---- 1. Basic Summary ----
glimpse(df)  # Structure of the data
summary(df)  # Summary statistics
skimr::skim(df)  # More detailed summary if skimr is installed

# ---- 2. Check Missing Values ----
colSums(is.na(df))  # Count NA values per column

# ---- 3. Visualize Distributions ----
df %>%
  select(water_temp_celcius, doxy_mg_L, ammonia_umol_kg, pH, chl_a_mg_L) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Value, fill = Variable)) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
  facet_wrap(~ Variable, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution of Water Quality Parameters")

# ---- 4. Correlation Analysis ----
df_cor <- df %>% select(water_temp_celcius, doxy_mg_L, ammonia_umol_kg, pH, chl_a_mg_L)# %>% na.omit()
cor_matrix <- cor(df_cor, use = "pairwise")

# Plot Correlation Matrix
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.cex = 0.8, addCoef.col = "black")

# ---- 5. Pairwise Scatter Plots ----
ggpairs(df_cor)

# ---- 6. Explore Package for Automated EDA ----
explore(df)  # Generates an interactive HTML report

# OR specific columns
df %>%
  select(water_temp_celcius, doxy_mg_L, ammonia_umol_kg, pH, chl_a_mg_L) %>%
  explore()

