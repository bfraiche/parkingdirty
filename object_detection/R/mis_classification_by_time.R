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

raw <- read_csv('raw_data/csvfile (4).csv',
                col_names = col_names)

raw %>%
  gather(key = 'threshold', value, -label, -image) %>% 
  mutate(value = if_else(value >= 1, 1, 0)) %>% 
  mutate(match = if_else(label == value, 1, 0)) %>% 
  group_by(threshold) %>% 
  summarize(tot_correct = sum(match),
            perc_correct = tot_correct / n()) %>% 
  arrange(desc(tot_correct))

head(raw$image)

out <- 
  raw %>% 
  mutate(obstacle_combined = `obstacle_25%` + obstacle_centerPoint) %>% 
  mutate_at(vars(starts_with("obstacle")), funs(if_else(.>=1, 1, 0))) %>% 
  mutate_at(vars(starts_with("obstacle")), funs(if_else(.==label, 1, 0))) %>% 
  mutate(datetime = str_extract(string = image, pattern = "(?<=blocked/).*(?= cam)"),
         date = str_trim(str_extract(datetime, pattern = ".*\\s")),
         time = str_trim(str_extract(datetime, pattern = "\\s.*")),
         hour = as.numeric(str_extract(time, "^.{2}"))) %>% 
  select(hour, obstacle_combined, label) %>% 
  group_by(hour, label) %>% 
  count(obstacle_combined) %>% 
  mutate(perc = n / sum(n)) %>% 
  mutate(n = case_when(obstacle_combined == 0 ~ -n,
                       TRUE ~ n),
         obstacle_combined = as.character(obstacle_combined))


scaleFactor <- max(out$n) / max(out$perc)

misclass_hour <- ggplot(out, aes(x = hour)) +
  geom_bar(aes(y = n, fill = obstacle_combined), stat = "identity")+
  geom_line(data = subset(out, obstacle_combined == "1"), 
            aes(y = perc * scaleFactor))+
  geom_text(data = subset(out, obstacle_combined == "1"), 
            aes(y = perc * scaleFactor + 7,
                label = round(perc, 2)))+
  scale_y_continuous(name="correct classification", sec.axis=sec_axis(~./scaleFactor, name="hourly percent correct classification"))+
  facet_grid(vars(label))

misclass_hour
