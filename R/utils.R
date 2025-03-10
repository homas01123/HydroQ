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


# Package ID: edi.1778.3 Cataloging System:https://pasta.edirepository.org.
# Data set title: OLIGOTREND, a global database of multi-decadal timeseries of chlorophyll-a and nutrient concentrations in inland and transitional waters, 1986-2023.
# Data set creator:  Camille Minaudo - University of Barcelona 
# Data set creator:  Xavier Benito-Granell - IRTA Sant Carles de la RÃ pita 
# Contact:  Camille Minaudo - Researcher University of Barcelona  - camille.minaudo@ub.edu
# Contact:  Xavier Benito-Granell - Researcher IRTA Sant Carles de la RÃ pita  - xavier.benito@irta.cat
# Stylesheet v2.14 for metadata conversion into program: John H. Porter, Univ. Virginia, jporter@virginia.edu      
# Uncomment the following lines to have R clear previous work, or set a working directory
# rm(list=ls())      

# setwd("C:/users/my_name/my_dir")       



# options(HTTPUserAgent="EDI_CodeGen")
# 
# 
# inUrl1  <- "https://pasta.lternet.edu/package/data/eml/edi/1778/3/c828f5056b9c46b8e120bc2c9406de05" 
# infile1 <- tempfile()
# try(download.file(inUrl1,infile1,method="curl",extra=paste0(' -A "',getOption("HTTPUserAgent"),'"')))
# if (is.na(file.size(infile1))) download.file(inUrl1,infile1,method="auto")
# 
# 
# dt1 <-read.csv(infile1,header=F 
#                ,skip=1
#                ,sep=","  
#                ,quot='"' 
#                , col.names=c(
#                  "Source",     
#                  "Link.to.data",     
#                  "Spatial.coverage",     
#                  "median_timeframe",     
#                  "n_total",     
#                  "n_estuary",     
#                  "n_lake",     
#                  "n_river",     
#                  "length_yrs",     
#                  "n_years",     
#                  "n_obs",     
#                  "yearly_n_obs_chla",     
#                  "n_obs_total"    ), check.names=TRUE)
# 
# unlink(infile1)
# 
# # Fix any interval or ratio columns mistakenly read in as nominal and nominal columns read as numeric or dates read as strings
# 
# if (class(dt1$Source)!="factor") dt1$Source<- as.factor(dt1$Source)
# if (class(dt1$Link.to.data)!="factor") dt1$Link.to.data<- as.factor(dt1$Link.to.data)
# if (class(dt1$Spatial.coverage)!="factor") dt1$Spatial.coverage<- as.factor(dt1$Spatial.coverage)
# if (class(dt1$median_timeframe)!="factor") dt1$median_timeframe<- as.factor(dt1$median_timeframe)
# if (class(dt1$n_total)=="factor") dt1$n_total <-as.numeric(levels(dt1$n_total))[as.integer(dt1$n_total) ]               
# if (class(dt1$n_total)=="character") dt1$n_total <-as.numeric(dt1$n_total)
# if (class(dt1$n_estuary)=="factor") dt1$n_estuary <-as.numeric(levels(dt1$n_estuary))[as.integer(dt1$n_estuary) ]               
# if (class(dt1$n_estuary)=="character") dt1$n_estuary <-as.numeric(dt1$n_estuary)
# if (class(dt1$n_lake)=="factor") dt1$n_lake <-as.numeric(levels(dt1$n_lake))[as.integer(dt1$n_lake) ]               
# if (class(dt1$n_lake)=="character") dt1$n_lake <-as.numeric(dt1$n_lake)
# if (class(dt1$n_river)=="factor") dt1$n_river <-as.numeric(levels(dt1$n_river))[as.integer(dt1$n_river) ]               
# if (class(dt1$n_river)=="character") dt1$n_river <-as.numeric(dt1$n_river)
# if (class(dt1$length_yrs)!="factor") dt1$length_yrs<- as.factor(dt1$length_yrs)
# if (class(dt1$n_years)!="factor") dt1$n_years<- as.factor(dt1$n_years)
# if (class(dt1$n_obs)!="factor") dt1$n_obs<- as.factor(dt1$n_obs)
# if (class(dt1$yearly_n_obs_chla)!="factor") dt1$yearly_n_obs_chla<- as.factor(dt1$yearly_n_obs_chla)
# if (class(dt1$n_obs_total)!="factor") dt1$n_obs_total<- as.factor(dt1$n_obs_total)
# 
# # Convert Missing Values to NA for non-dates
# 
# 
# 
# # Here is the structure of the input data frame:
# str(dt1)                            
# attach(dt1)                            
# # The analyses below are basic descriptions of the variables. After testing, they should be replaced.                 
# 
# summary(Source)
# summary(Link.to.data)
# summary(Spatial.coverage)
# summary(median_timeframe)
# summary(n_total)
# summary(n_estuary)
# summary(n_lake)
# summary(n_river)
# summary(length_yrs)
# summary(n_years)
# summary(n_obs)
# summary(yearly_n_obs_chla)
# summary(n_obs_total) 
# # Get more details on character variables
# 
# summary(as.factor(dt1$Source)) 
# summary(as.factor(dt1$Link.to.data)) 
# summary(as.factor(dt1$Spatial.coverage)) 
# summary(as.factor(dt1$median_timeframe)) 
# summary(as.factor(dt1$length_yrs)) 
# summary(as.factor(dt1$n_years)) 
# summary(as.factor(dt1$n_obs)) 
# summary(as.factor(dt1$yearly_n_obs_chla)) 
# summary(as.factor(dt1$n_obs_total))
# detach(dt1)               
# 
# 
# 
# inUrl2  <- "https://pasta.lternet.edu/package/data/eml/edi/1778/3/cc48f89ff50a51a6e9dbf9e35fc16c20" 
# infile2 <- tempfile()
# try(download.file(inUrl2,infile2,method="curl",extra=paste0(' -A "',getOption("HTTPUserAgent"),'"')))
# if (is.na(file.size(infile2))) download.file(inUrl2,infile2,method="auto")
# 
# 
# dt2 <-read.csv(infile2,header=F 
#                ,skip=1
#                ,sep=","  
#                ,quot='"' 
#                , col.names=c(
#                  "basin",     
#                  "ecosystem",     
#                  "id",     
#                  "variable",     
#                  "date",     
#                  "value",     
#                  "flag"    ), check.names=TRUE)
# 
# unlink(infile2)
# 
# # Fix any interval or ratio columns mistakenly read in as nominal and nominal columns read as numeric or dates read as strings
# 
# if (class(dt2$basin)!="factor") dt2$basin<- as.factor(dt2$basin)
# if (class(dt2$ecosystem)!="factor") dt2$ecosystem<- as.factor(dt2$ecosystem)
# if (class(dt2$id)!="factor") dt2$id<- as.factor(dt2$id)
# if (class(dt2$variable)!="factor") dt2$variable<- as.factor(dt2$variable)                                   
# # attempting to convert dt2$date dateTime string to R date structure (date or POSIXct)                                
# tmpDateFormat<-"%Y-%m-%d"
# tmp2date<-as.Date(dt2$date,format=tmpDateFormat)
# # Keep the new dates only if they all converted correctly
# if(nrow(dt2[dt2$date != "",]) == length(tmp2date[!is.na(tmp2date)])){dt2$date <- tmp2date } else {print("Date conversion failed for dt2$date. Please inspect the data and do the date conversion yourself.")}                                                                    
# 
# if (class(dt2$value)=="factor") dt2$value <-as.numeric(levels(dt2$value))[as.integer(dt2$value) ]               
# if (class(dt2$value)=="character") dt2$value <-as.numeric(dt2$value)
# if (class(dt2$flag)!="factor") dt2$flag<- as.factor(dt2$flag)
# 
# # Convert Missing Values to NA for non-dates
# 
# 
# 
# # Here is the structure of the input data frame:
# str(dt2)                            
# attach(dt2)                            
# # The analyses below are basic descriptions of the variables. After testing, they should be replaced.                 
# 
# summary(basin)
# summary(ecosystem)
# summary(id)
# summary(variable)
# summary(date)
# summary(value)
# summary(flag) 
# # Get more details on character variables
# 
# summary(as.factor(dt2$basin)) 
# summary(as.factor(dt2$ecosystem)) 
# summary(as.factor(dt2$id)) 
# summary(as.factor(dt2$variable)) 
# summary(as.factor(dt2$flag))
# detach(dt2)               
# 
# 
# 
# inUrl3  <- "https://pasta.lternet.edu/package/data/eml/edi/1778/3/4fb9081418e2f8112f094bb284197dde" 
# infile3 <- tempfile()
# try(download.file(inUrl3,infile3,method="curl",extra=paste0(' -A "',getOption("HTTPUserAgent"),'"')))
# if (is.na(file.size(infile3))) download.file(inUrl3,infile3,method="auto")
# 
# 
# dt3 <-read.csv(infile3,header=F 
#                ,skip=1
#                ,sep=","  
#                ,quot='"' 
#                , col.names=c(
#                  "X",     
#                  "Y",     
#                  "uniqID",     
#                  "basin",     
#                  "ecsystm"    ), check.names=TRUE)
# 
# unlink(infile3)
# 
# # Fix any interval or ratio columns mistakenly read in as nominal and nominal columns read as numeric or dates read as strings
# 
# if (class(dt3$X)=="factor") dt3$X <-as.numeric(levels(dt3$X))[as.integer(dt3$X) ]               
# if (class(dt3$X)=="character") dt3$X <-as.numeric(dt3$X)
# if (class(dt3$Y)=="factor") dt3$Y <-as.numeric(levels(dt3$Y))[as.integer(dt3$Y) ]               
# if (class(dt3$Y)=="character") dt3$Y <-as.numeric(dt3$Y)
# if (class(dt3$uniqID)!="factor") dt3$uniqID<- as.factor(dt3$uniqID)
# if (class(dt3$basin)!="factor") dt3$basin<- as.factor(dt3$basin)
# if (class(dt3$ecsystm)!="factor") dt3$ecsystm<- as.factor(dt3$ecsystm)
# 
# # Convert Missing Values to NA for non-dates
# 
# 
# 
# # Here is the structure of the input data frame:
# str(dt3)                            
# attach(dt3)                            
# # The analyses below are basic descriptions of the variables. After testing, they should be replaced.                 
# 
# summary(X)
# summary(Y)
# summary(uniqID)
# summary(basin)
# summary(ecsystm) 
# # Get more details on character variables
# 
# summary(as.factor(dt3$uniqID)) 
# summary(as.factor(dt3$basin)) 
# summary(as.factor(dt3$ecsystm))
# detach(dt3)               
# 
# 
# 
# inUrl4  <- "https://pasta.lternet.edu/package/data/eml/edi/1778/3/00446b372fabf8e97bb23c2d47482a0d" 
# infile4 <- tempfile()
# try(download.file(inUrl4,infile4,method="curl",extra=paste0(' -A "',getOption("HTTPUserAgent"),'"')))
# if (is.na(file.size(infile4))) download.file(inUrl4,infile4,method="auto")
# 
# 
# dt4 <-read.csv(infile4,header=F 
#                ,skip=1
#                ,sep=","  
#                ,quot='"' 
#                , col.names=c(
#                  "id",     
#                  "ecosystem",     
#                  "basin",     
#                  "variable",     
#                  "year_start",     
#                  "sens_slope",     
#                  "sens_slope.Z",     
#                  "n",     
#                  "avg",     
#                  "sd",     
#                  "CV",     
#                  "avg_summer",     
#                  "mean_1st_half",     
#                  "mean_2nd_half",     
#                  "mean_1st_half_summer",     
#                  "mean_2nd_half_summer",     
#                  "p_neg",     
#                  "p_pos",     
#                  "spearman",     
#                  "pearson",     
#                  "isBP",     
#                  "BPpval",     
#                  "slope_seg1",     
#                  "slope_seg2",     
#                  "BP",     
#                  "BPconf",     
#                  "seg_type"    ), check.names=TRUE)
# 
# unlink(infile4)
# 
# # Fix any interval or ratio columns mistakenly read in as nominal and nominal columns read as numeric or dates read as strings
# 
# if (class(dt4$id)!="factor") dt4$id<- as.factor(dt4$id)
# if (class(dt4$ecosystem)!="factor") dt4$ecosystem<- as.factor(dt4$ecosystem)
# if (class(dt4$basin)!="factor") dt4$basin<- as.factor(dt4$basin)
# if (class(dt4$variable)!="factor") dt4$variable<- as.factor(dt4$variable)
# if (class(dt4$year_start)=="factor") dt4$year_start <-as.numeric(levels(dt4$year_start))[as.integer(dt4$year_start) ]               
# if (class(dt4$year_start)=="character") dt4$year_start <-as.numeric(dt4$year_start)
# if (class(dt4$sens_slope)=="factor") dt4$sens_slope <-as.numeric(levels(dt4$sens_slope))[as.integer(dt4$sens_slope) ]               
# if (class(dt4$sens_slope)=="character") dt4$sens_slope <-as.numeric(dt4$sens_slope)
# if (class(dt4$sens_slope.Z)=="factor") dt4$sens_slope.Z <-as.numeric(levels(dt4$sens_slope.Z))[as.integer(dt4$sens_slope.Z) ]               
# if (class(dt4$sens_slope.Z)=="character") dt4$sens_slope.Z <-as.numeric(dt4$sens_slope.Z)
# if (class(dt4$n)=="factor") dt4$n <-as.numeric(levels(dt4$n))[as.integer(dt4$n) ]               
# if (class(dt4$n)=="character") dt4$n <-as.numeric(dt4$n)
# if (class(dt4$avg)=="factor") dt4$avg <-as.numeric(levels(dt4$avg))[as.integer(dt4$avg) ]               
# if (class(dt4$avg)=="character") dt4$avg <-as.numeric(dt4$avg)
# if (class(dt4$sd)=="factor") dt4$sd <-as.numeric(levels(dt4$sd))[as.integer(dt4$sd) ]               
# if (class(dt4$sd)=="character") dt4$sd <-as.numeric(dt4$sd)
# if (class(dt4$CV)=="factor") dt4$CV <-as.numeric(levels(dt4$CV))[as.integer(dt4$CV) ]               
# if (class(dt4$CV)=="character") dt4$CV <-as.numeric(dt4$CV)
# if (class(dt4$avg_summer)=="factor") dt4$avg_summer <-as.numeric(levels(dt4$avg_summer))[as.integer(dt4$avg_summer) ]               
# if (class(dt4$avg_summer)=="character") dt4$avg_summer <-as.numeric(dt4$avg_summer)
# if (class(dt4$mean_1st_half)=="factor") dt4$mean_1st_half <-as.numeric(levels(dt4$mean_1st_half))[as.integer(dt4$mean_1st_half) ]               
# if (class(dt4$mean_1st_half)=="character") dt4$mean_1st_half <-as.numeric(dt4$mean_1st_half)
# if (class(dt4$mean_2nd_half)=="factor") dt4$mean_2nd_half <-as.numeric(levels(dt4$mean_2nd_half))[as.integer(dt4$mean_2nd_half) ]               
# if (class(dt4$mean_2nd_half)=="character") dt4$mean_2nd_half <-as.numeric(dt4$mean_2nd_half)
# if (class(dt4$mean_1st_half_summer)=="factor") dt4$mean_1st_half_summer <-as.numeric(levels(dt4$mean_1st_half_summer))[as.integer(dt4$mean_1st_half_summer) ]               
# if (class(dt4$mean_1st_half_summer)=="character") dt4$mean_1st_half_summer <-as.numeric(dt4$mean_1st_half_summer)
# if (class(dt4$mean_2nd_half_summer)=="factor") dt4$mean_2nd_half_summer <-as.numeric(levels(dt4$mean_2nd_half_summer))[as.integer(dt4$mean_2nd_half_summer) ]               
# if (class(dt4$mean_2nd_half_summer)=="character") dt4$mean_2nd_half_summer <-as.numeric(dt4$mean_2nd_half_summer)
# if (class(dt4$p_neg)=="factor") dt4$p_neg <-as.numeric(levels(dt4$p_neg))[as.integer(dt4$p_neg) ]               
# if (class(dt4$p_neg)=="character") dt4$p_neg <-as.numeric(dt4$p_neg)
# if (class(dt4$p_pos)=="factor") dt4$p_pos <-as.numeric(levels(dt4$p_pos))[as.integer(dt4$p_pos) ]               
# if (class(dt4$p_pos)=="character") dt4$p_pos <-as.numeric(dt4$p_pos)
# if (class(dt4$spearman)=="factor") dt4$spearman <-as.numeric(levels(dt4$spearman))[as.integer(dt4$spearman) ]               
# if (class(dt4$spearman)=="character") dt4$spearman <-as.numeric(dt4$spearman)
# if (class(dt4$pearson)=="factor") dt4$pearson <-as.numeric(levels(dt4$pearson))[as.integer(dt4$pearson) ]               
# if (class(dt4$pearson)=="character") dt4$pearson <-as.numeric(dt4$pearson)
# if (class(dt4$isBP)!="factor") dt4$isBP<- as.factor(dt4$isBP)
# if (class(dt4$BPpval)=="factor") dt4$BPpval <-as.numeric(levels(dt4$BPpval))[as.integer(dt4$BPpval) ]               
# if (class(dt4$BPpval)=="character") dt4$BPpval <-as.numeric(dt4$BPpval)
# if (class(dt4$slope_seg1)=="factor") dt4$slope_seg1 <-as.numeric(levels(dt4$slope_seg1))[as.integer(dt4$slope_seg1) ]               
# if (class(dt4$slope_seg1)=="character") dt4$slope_seg1 <-as.numeric(dt4$slope_seg1)
# if (class(dt4$slope_seg2)=="factor") dt4$slope_seg2 <-as.numeric(levels(dt4$slope_seg2))[as.integer(dt4$slope_seg2) ]               
# if (class(dt4$slope_seg2)=="character") dt4$slope_seg2 <-as.numeric(dt4$slope_seg2)
# if (class(dt4$BP)=="factor") dt4$BP <-as.numeric(levels(dt4$BP))[as.integer(dt4$BP) ]               
# if (class(dt4$BP)=="character") dt4$BP <-as.numeric(dt4$BP)
# if (class(dt4$BPconf)=="factor") dt4$BPconf <-as.numeric(levels(dt4$BPconf))[as.integer(dt4$BPconf) ]               
# if (class(dt4$BPconf)=="character") dt4$BPconf <-as.numeric(dt4$BPconf)
# if (class(dt4$seg_type)!="factor") dt4$seg_type<- as.factor(dt4$seg_type)
# 
# # Convert Missing Values to NA for non-dates
# 
# dt4$sens_slope <- ifelse((trimws(as.character(dt4$sens_slope))==trimws("NA")),NA,dt4$sens_slope)               
# suppressWarnings(dt4$sens_slope <- ifelse(!is.na(as.numeric("NA")) & (trimws(as.character(dt4$sens_slope))==as.character(as.numeric("NA"))),NA,dt4$sens_slope))
# dt4$sens_slope.Z <- ifelse((trimws(as.character(dt4$sens_slope.Z))==trimws("NA")),NA,dt4$sens_slope.Z)               
# suppressWarnings(dt4$sens_slope.Z <- ifelse(!is.na(as.numeric("NA")) & (trimws(as.character(dt4$sens_slope.Z))==as.character(as.numeric("NA"))),NA,dt4$sens_slope.Z))
# dt4$CV <- ifelse((trimws(as.character(dt4$CV))==trimws("Inf")),NA,dt4$CV)               
# suppressWarnings(dt4$CV <- ifelse(!is.na(as.numeric("Inf")) & (trimws(as.character(dt4$CV))==as.character(as.numeric("Inf"))),NA,dt4$CV))
# dt4$slope_seg1 <- ifelse((trimws(as.character(dt4$slope_seg1))==trimws("NA")),NA,dt4$slope_seg1)               
# suppressWarnings(dt4$slope_seg1 <- ifelse(!is.na(as.numeric("NA")) & (trimws(as.character(dt4$slope_seg1))==as.character(as.numeric("NA"))),NA,dt4$slope_seg1))
# dt4$slope_seg2 <- ifelse((trimws(as.character(dt4$slope_seg2))==trimws("NA")),NA,dt4$slope_seg2)               
# suppressWarnings(dt4$slope_seg2 <- ifelse(!is.na(as.numeric("NA")) & (trimws(as.character(dt4$slope_seg2))==as.character(as.numeric("NA"))),NA,dt4$slope_seg2))
# dt4$BP <- ifelse((trimws(as.character(dt4$BP))==trimws("NA")),NA,dt4$BP)               
# suppressWarnings(dt4$BP <- ifelse(!is.na(as.numeric("NA")) & (trimws(as.character(dt4$BP))==as.character(as.numeric("NA"))),NA,dt4$BP))
# dt4$BPconf <- ifelse((trimws(as.character(dt4$BPconf))==trimws("NA")),NA,dt4$BPconf)               
# suppressWarnings(dt4$BPconf <- ifelse(!is.na(as.numeric("NA")) & (trimws(as.character(dt4$BPconf))==as.character(as.numeric("NA"))),NA,dt4$BPconf))
# 
# 
# # Here is the structure of the input data frame:
# str(dt4)                            
# attach(dt4)                            
# # The analyses below are basic descriptions of the variables. After testing, they should be replaced.                 
# 
# summary(id)
# summary(ecosystem)
# summary(basin)
# summary(variable)
# summary(year_start)
# summary(sens_slope)
# summary(sens_slope.Z)
# summary(n)
# summary(avg)
# summary(sd)
# summary(CV)
# summary(avg_summer)
# summary(mean_1st_half)
# summary(mean_2nd_half)
# summary(mean_1st_half_summer)
# summary(mean_2nd_half_summer)
# summary(p_neg)
# summary(p_pos)
# summary(spearman)
# summary(pearson)
# summary(isBP)
# summary(BPpval)
# summary(slope_seg1)
# summary(slope_seg2)
# summary(BP)
# summary(BPconf)
# summary(seg_type) 
# # Get more details on character variables
# 
# summary(as.factor(dt4$id)) 
# summary(as.factor(dt4$ecosystem)) 
# summary(as.factor(dt4$basin)) 
# summary(as.factor(dt4$variable)) 
# summary(as.factor(dt4$isBP)) 
# summary(as.factor(dt4$seg_type))
# detach(dt4)               
# 
# 
# # Load necessary libraries
# library(dplyr)
# library(tidyr)
# 
# # Assuming your data frame is named df
# # Filter the data for the required variables and reshape the data
# result <- dt4 %>%
#   filter(variable %in% c("wtemp", "o2", "ph", "chla")) %>%
#   select(id, year_start, variable, avg) %>%
#   pivot_wider(names_from = variable, values_from = avg) %>%
#   arrange(id, year_start)
# 
# # View the result
# print(result)





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

