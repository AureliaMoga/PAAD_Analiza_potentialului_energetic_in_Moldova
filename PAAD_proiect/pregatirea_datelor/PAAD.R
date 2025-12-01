library(jsonlite)
library(dplyr)
library(tidyr)
library(stringr)

# Folderul unde ai fișierele
base_dir   <- path.expand("~/Desktop/PAAD/date")
nasa_path  <- file.path(base_dir, "NASA.json")
stat_path  <- file.path(base_dir, "statisticaMD.json")

# =========================
# 1) NASA.json -> tabel lunar
# =========================

nasa_raw <- fromJSON(nasa_path)

params <- nasa_raw$properties$parameter

nasa_long <- lapply(names(params), function(par) {
  v <- unlist(params[[par]])
  data.frame(
    ym    = names(v),
    var   = par,
    value = as.numeric(v),
    stringsAsFactors = FALSE
  )
}) |> bind_rows()

# eliminăm lunile "13" (anual) și valorile -999 (no data)
nasa_long <- nasa_long |>
  filter(substr(ym, 5, 6) != "13",
         value != -999)

# adăugăm YEAR + MONTH
nasa_long <- nasa_long |>
  mutate(
    year  = as.integer(substr(ym, 1, 4)),
    month = as.integer(substr(ym, 5, 6))
  )

# wide: o coloană pe fiecare parametru (GHI, vânt, etc.)
nasa_df <- nasa_long |>
  select(year, month, var, value) |>
  pivot_wider(names_from = var, values_from = value)

# opțional: vezi ce coloane ai (să confirmi GHI și vântul)
print(names(nasa_df))

# =========================
# 2) statisticaMD.json -> tabel lunar cu indicatori denumiți clar
# =========================

stat_raw <- fromJSON(stat_path)

# coloanele sunt:
# - stat_raw$data$key    : listă de vectori c(indicator, an, lună)
# - stat_raw$data$values : listă cu o singură valoare numerică pe fiecare rând

keys   <- stat_raw$data$key
values <- stat_raw$data$values

stat_data <- tibble::tibble(
  indicator_code = sapply(keys, `[`, 1),
  year           = as.integer(sapply(keys, `[`, 2)),
  month          = as.integer(sapply(keys, `[`, 3)),
  value          = as.numeric(unlist(values))
)

# mapăm codurile 1,2,3,... la denumiri clare
stat_data <- stat_data |>
  mutate(
    indicator = dplyr::recode(
      indicator_code,
      "1" = "producere",
      "2" = "import",
      "3" = "procurat_din_alte_surse",
      "4" = "variatia_stocurilor",
      "5" = "export",
      "6" = "consum_final_brut",
      "7" = "consum_sector_rezidential",
      .default = paste0("indicator_", indicator_code)
    )
  )

# din long (year, month, indicator, value) -> wide (o coloană per indicator)
stat_df <- stat_data |>
  select(year, month, indicator, value) |>
  tidyr::pivot_wider(names_from = indicator, values_from = value)

head(stat_df)

# =========================
# 3) Unim NASA + Statistica după (year, month)
# =========================

df_all <- full_join(nasa_df, stat_df, by = c("year", "month"))

# (bonus) coloană date pentru analize ulterioare
df_all <- df_all |>
  mutate(
    date = as.Date(sprintf("%04d-%02d-01", year, month))
  )

# =========================
# 4) Energia solară estimată (PV)
# =========================

ghi_col <- "ALLSKY_SFC_SW_DWN"  # GHI din NASA.json (vezi în names(nasa_df))

panel_area_m2 <- 20      # suprafață totală panouri (m²) – poți schimba
panel_eff     <- 0.18    # randament panouri (18%)
system_losses <- 0.15    # pierderi sistem (15%)

df_all <- df_all |>
  mutate(
    pv_energy_kwh_day = ifelse(
      is.na(.data[[ghi_col]]),
      NA_real_,
      .data[[ghi_col]] * panel_area_m2 * panel_eff * (1 - system_losses)
    )
  )

# =========================
# 5) Energia eoliană estimată
# =========================

# vezi cum se numește exact coloana de vânt:
# print(names(nasa_df))
wind_col      <- "WS50M"   # dacă în names(nasa_df) este alt nume, îl pui aici
rotor_radius  <- 40        # rază rotor (m) – exemplu
turbine_eff   <- 0.40      # randament global turbină
air_density   <- 1.225     # densitatea aerului (kg/m3)
hours_per_day <- 24

df_all <- df_all |>
  mutate(
    rotor_area_m2 = pi * rotor_radius^2,
    wind_energy_kwh_day = ifelse(
      is.na(.data[[wind_col]]),
      NA_real_,
      0.5 * air_density * rotor_area_m2 * (.data[[wind_col]]^3) *
        turbine_eff * (hours_per_day * 3600) / 3.6e6
    )
  )

# =========================
# 6) Salvăm setul de date complet
# =========================

write.csv(
  df_all,
  file.path(base_dir, "dataset_complet_energie.csv"),
  row.names = FALSE
)
