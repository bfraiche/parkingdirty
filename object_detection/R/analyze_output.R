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
  mutate(false_pos = if_else(match == 0 & label == 0, 1, 0),
         false_neg = if_else(match == 0 & label == 1, 1, 0)) %>%
  group_by(threshold) %>%
  summarize(tot_correct = sum(match),
            tot_false_pos = sum(false_pos),
            tot_false_neg = sum(false_neg),
            perc_correct = tot_correct / n()) %>% 
  arrange(desc(tot_correct))

print(output)

