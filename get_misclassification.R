library(dplyr)
library(readr)
library(stringr)
library(ggplot2)
library(tidyr)

col_names <- c('image','obstacle_10%',
               'obstacle_15%','obstacle_20%','obstacle_25%',
               'obstacle_30%','obstacle_35%',
               'obstacle_40%','obstacle_45%',
               'obstacle_50%', 'obstacle_centerPoint','obstacle_bikes','label')

raw <- read_csv(commandArgs(trailingOnly = TRUE),
                col_names = col_names)

output <- 
  raw %>%
  gather(key = 'threshold', value, -label, -image) %>% 
  mutate(value = if_else(value >= 1, 1, 0)) %>% 
  mutate(match = if_else(label == value, 1, 0)) %>% 
  filter(match == 0)

print(output)

