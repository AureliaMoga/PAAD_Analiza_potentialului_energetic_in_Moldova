library(tidyverse)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(ggplot2)

setwd("~/Desktop/PAAD/Intrebarea1")

citire_power <- function(file, regiune) {
  read_csv(file, skip = 9, show_col_types = FALSE) %>%   # sari peste primele 9 linii
    select(-PARAMETER, -ANN) %>%                        # scoatem coloanele inutile
    pivot_longer(
      cols = -YEAR,
      names_to  = "MONTH",
      values_to = regiune                               # numele coloanei = regiunea
    )
}

citire_solar <- function(file, regiune) {
  read_csv(file, skip = 9, show_col_types = FALSE) %>%
    select(-PARAMETER, -ANN) %>% 
    pivot_longer(
      cols = -YEAR,
      names_to  = "MONTH",
      values_to = regiune
    )
}

nord   <- citire_power("Nord.csv",   "Nord")
centru <- citire_power("Centru.csv", "Centru")
sud    <- citire_power("Sud.csv",    "Sud")

solar_nord   <- citire_solar("solar_nord.csv",   "Solar_Nord")
solar_centru <- citire_solar("solar_centru.csv", "Solar_Centru")
solar_sud    <- citire_solar("solar_sud.csv",    "Solar_Sud")


date_final <- nord %>%
  left_join(centru, by = c("YEAR", "MONTH")) %>%
  left_join(sud,    by = c("YEAR", "MONTH"))

date_solar <- solar_nord %>%
  left_join(solar_centru, by = c("YEAR", "MONTH")) %>%
  left_join(solar_sud,    by = c("YEAR", "MONTH"))


# eliminăm ultimele  luni incomplete
# Curățare solar — scoatem rândurile cu -999
date_final <- date_final %>% 
  filter(Nord   != -999,
         Centru != -999,
         Sud    != -999)

date_solar <- date_solar %>% 
  filter(Solar_Nord   != -999,
         Solar_Centru != -999,
         Solar_Sud    != -999)

# dataset COMUN: vânt + solar pe aceleași YEAR, MONTH
date_all <- inner_join(date_final, date_solar, by = c("YEAR", "MONTH"))

# salvăm separat vânt și solar (opțional, dar util)
write_csv(date_final, "Vant_2015_2025_regiuni.csv")
write_csv(date_solar, "Solar_2015_2025_regiuni.csv")

# salvăm fișierul mare cu toate datele
write_csv(date_all, "Vant_Solar_2015_2025_regiuni.csv")


#Vizualizarea

medii_regiuni <- date_final %>% 
  summarise(
    Nord   = mean(Nord,   na.rm = TRUE),
    Centru = mean(Centru, na.rm = TRUE),
    Sud    = mean(Sud,    na.rm = TRUE)
  ) %>%
  pivot_longer(cols = everything(),
               names_to = "Regiune",
               values_to = "Viteza_medie")

medii_regiuni



# Țara Moldova în format sf
moldova <- ne_countries(country = "Moldova",
                        scale = "medium",
                        returnclass = "sf")

# Bbox (dreptunghiul care cuprinde Moldova)
bb <- st_bbox(moldova)

# Grid cu 3 benzi orizontale (Nord, Centru, Sud)
grid3 <- st_make_grid(moldova, n = c(1, 3))  # 1 pe X, 3 pe Y

# Intersecția dintre Moldova și grid -> 3 poligoane
zone3 <- st_intersection(moldova, grid3) %>%
  st_as_sf() %>%
  mutate(Regiune = c("Sud", "Centru", "Nord"))  # de jos în sus


# Atașăm viteza medie pe regiune
zone3_wind <- zone3 %>%
  left_join(medii_regiuni, by = "Regiune")



ggplot(zone3_wind) +
  geom_sf(aes(fill = Regiune), color = "white", linewidth = 0.4) +
  geom_sf_text(aes(label = round(Viteza_medie, 2)),
               color = "white", fontface = "bold", size = 4) +
  scale_fill_manual(
    values = c(
      "Nord"   = "#0033A0",   # albastru
      "Centru" = "#FFD700",   # galben
      "Sud"    = "#D00000"    # roșu
    ),
    name = "Regiune"
  ) +
  theme_minimal() +
  labs(
    title = "Potențialul eolian pe regiuni\nRepublica Moldova (2015–2025)",
    subtitle = "Viteza medie a vântului la 50 m",
    caption = "Sursă: NASA POWER, prelucrare proprie"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "right"
  )

# medii solare pe regiuni
medii_solar <- date_solar %>%
  summarise(
    Nord   = mean(Solar_Nord,   na.rm = TRUE),
    Centru = mean(Solar_Centru, na.rm = TRUE),
    Sud    = mean(Solar_Sud,    na.rm = TRUE)
  ) %>%
  pivot_longer(cols = everything(),
               names_to = "Regiune",
               values_to = "Irradianta_medie")

medii_solar

# atașăm mediile solare pe aceleași 3 zone geografice
zone3_solar <- zone3 %>%
  left_join(medii_solar, by = "Regiune")

# hartă solară (stil drapel, la fel ca vântul)
ggplot(zone3_solar) +
  geom_sf(aes(fill = Regiune), color = "white", linewidth = 0.4) +
  geom_sf_text(aes(label = round(Irradianta_medie, 2)),
               color = "white", fontface = "bold", size = 4) +
  scale_fill_manual(
    values = c(
      "Nord"   = "#0033A0",   # albastru
      "Centru" = "#FFD700",   # galben
      "Sud"    = "#D00000"    # roșu
    ),
    name = "Regiune"
  ) +
  theme_minimal() +
  labs(
    title = "Potențialul solar pe regiuni\nRepublica Moldova (2015–2025)",
    subtitle = "Iradianță solară medie (All Sky Surface Shortwave Downward Irradiance)",
    caption = "Sursă: NASA POWER, prelucrare proprie"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "right"
  )


