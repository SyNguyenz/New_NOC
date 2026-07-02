#!/usr/bin/env Rscript
# synth/simulate_mixtures.R  —  Task 2.2
#
# Simulate STR DNA mixtures using simDNAmixtures (Kruijver & Bright 2022).
# Same simulator as deepNoC (Taylor & Humphries 2024); we use only the
# peak-info layer (no GAN/EPG) because our model consumes peak tokens.
#
# Protocol:
#   1. Read donor_genotypes.csv (45 known donors, 24 loci)
#   2. Read spec.json: list of {combo, n_mixtures, noc, seed_offset}
#   3. For each spec entry: sample n_mixtures with random Mx and template
#      from PROVEDIt-matched ranges
#   4. Write output CSV: sample_id, locus, allele, height, donor_ids, noc, ratios
#
# Usage (called from generate_dataset.py via subprocess):
#   Rscript synth/simulate_mixtures.R --spec <spec.json> --out <out.csv> --seed <int>
#
# Calibration note:
#   model params = gf_configuration() defaults (published GlobalFiler params).
#   Template range [0.003, 0.76] ng matched to PROVEDIt GF29cycles dataset.
#   Full lab calibration is out of scope for this workshop paper.

suppressPackageStartupMessages({
  library(simDNAmixtures)
  library(jsonlite)
  library(dplyr)
})

# ── CLI args ─────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- which(args == flag)
  if (length(i) == 0) return(default)
  args[i + 1]
}

spec_path <- get_arg("--spec")
out_path  <- get_arg("--out")
seed_val  <- as.integer(get_arg("--seed", "42"))

if (is.null(spec_path) || is.null(out_path)) {
  stop("Usage: Rscript simulate_mixtures.R --spec <spec.json> --out <out.csv> [--seed <int>]")
}

set.seed(seed_val)
cat(sprintf("simDNAmixtures %s | seed=%d\n",
            packageVersion("simDNAmixtures"), seed_val))

# ── Load GF configuration ────────────────────────────────────────────────────
cfg <- gf_configuration()

# model_settings = cfg$log_normal_settings
# (already contains: locus_names, detection_threshold, c2_prior, LSAE_variance_prior,
#  size_regression, stutter_model, stutter_variability, degradation_parameter_cap)
model_settings <- cfg$log_normal_settings
if (is.null(model_settings$degradation_parameter_cap))
  model_settings$degradation_parameter_cap <- cfg$degradation_parameter_cap

cat("stutter_model present:", !is.null(model_settings$stutter_model), "\n")
cat("stutter_variability present:", !is.null(model_settings$stutter_variability), "\n")

# sampling_params = model_settings + PROVEDIt-matched template/degradation ranges
# Template range: 0.003-0.76 ng (from extract_metadata.py on GF29cycles)
sampling_params <- model_settings
# Template in simDNAmixtures internal units (calibrated to GF at Forensic Sci SA).
# NOT in nanograms — the model's c2_prior [8.45, 1.746] gives c2 ~ 14.8.
# Peak height ~ c2 * template * allele_proportion.
# For H ~ 100-8000 RFU: template range ~ [10, 600] works.
# This covers PROVEDIt GF29cycles range (templates from near-baseline to near-saturation).
sampling_params$min_template      <- 10.0    # near-baseline / very low template
sampling_params$max_template      <- 600.0   # near-saturation / high template
sampling_params$degradation_shape <- 1.0
sampling_params$degradation_scale <- 0.003   # mild degradation, cap=0.01

MODEL_LOCI <- model_settings$locus_names   # 22 autosomal (no DYS391, Yindel)
cat("Model loci (", length(MODEL_LOCI), "):", paste(MODEL_LOCI, collapse=", "), "\n")

# ── Load donor genotypes ─────────────────────────────────────────────────────
geno_arg  <- get_arg("--geno")
geno_path <- if (!is.null(geno_arg)) geno_arg else
  file.path(dirname(dirname(spec_path)), "data", "synth", "donor_genotypes.csv")
geno_df <- read.csv(geno_path, stringsAsFactors = FALSE)
cat("Genotypes loaded:", nrow(geno_df), "rows\n")

# Build lookup: donor_id -> data.frame(Sample Name, Locus, Allele1, Allele2)
# Only keep MODEL_LOCI (drop AMEL, DYS391, Yindel for simulation)
build_genotype_df <- function(donor_id) {
  d <- geno_df[geno_df$donor_id == donor_id & geno_df$locus %in% MODEL_LOCI, ]
  if (nrow(d) == 0) return(NULL)
  df <- data.frame(
    `Sample Name` = rep(as.character(donor_id), nrow(d)),
    Locus   = d$locus,
    Allele1 = ifelse(is.na(d$allele1), "14", as.character(d$allele1)),  # fallback
    Allele2 = ifelse(is.na(d$allele2), "14", as.character(d$allele2)),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  # Keep only loci the model knows about
  df <- df[df$Locus %in% MODEL_LOCI, ]
  df
}

# Pre-build all donor genotype DFs
all_donor_ids <- unique(geno_df$donor_id)
donor_geno_cache <- lapply(setNames(all_donor_ids, all_donor_ids), build_genotype_df)

# ── Mixture ratio sampling ───────────────────────────────────────────────────
# Sample Dirichlet-like mixture ratios
# NOC=2: use 1:1, 1:2, 1:3, 1:4, 1:5 with equal probability
# NOC>2: sample from Dirichlet(alpha=1) → normalize to sum=1
sample_ratios <- function(noc) {
  if (noc == 2) {
    preset <- sample(c(1,2,3,4,5), 1)
    r <- c(1, preset); r <- r / sum(r)
  } else {
    r <- rgamma(noc, shape = 1, rate = 1)
    r <- r / sum(r)
  }
  sort(r, decreasing = TRUE)  # major contributor first
}

# ── Simulation loop ──────────────────────────────────────────────────────────
spec <- fromJSON(spec_path)
cat("Spec entries:", length(spec$combo), "\n")

all_rows <- list()
n_simulated <- 0L

for (entry_i in seq_len(nrow(spec))) {
  combo      <- as.integer(unlist(spec$combo[entry_i]))
  n_mix      <- as.integer(spec$n_mixtures[entry_i])
  noc        <- length(combo)
  seed_off   <- if (!is.null(spec$seed_offset)) as.integer(spec$seed_offset[entry_i]) else 0L

  set.seed(seed_val + seed_off + entry_i * 1000L)

  cat(sprintf("  combo=%s  noc=%d  n=%d\n",
              paste(combo, collapse="+"), noc, n_mix))

  # Check all donors have genotype data
  missing_donors <- combo[!combo %in% names(donor_geno_cache) |
                           sapply(combo, function(d) is.null(donor_geno_cache[[as.character(d)]]))]
  if (length(missing_donors) > 0) {
    cat(sprintf("    WARNING: missing genotypes for donors %s, skipping\n",
                paste(missing_donors, collapse=",")))
    next
  }

  for (mix_i in seq_len(n_mix)) {
    tryCatch({
      # Sample mixture ratios
      ratios <- sample_ratios(noc)

      # Build model (samples template + degradation + c2 + LSAE)
      model <- sample_log_normal_model(
        number_of_contributors = noc,
        sampling_parameters    = sampling_params,
        model_settings         = model_settings
      )

      # Apply mixture ratios by scaling template
      # simDNAmixtures uses 'template' proportional to DNA amount
      total_template <- sum(model$template)
      model$template <- ratios * total_template

      # Build contributor genotype list
      genotypes <- lapply(combo, function(d) donor_geno_cache[[as.character(d)]])

      # Simulate peaks
      sample_name <- sprintf("synth_%05d", n_simulated + mix_i)
      mixture_df <- sample_mixture_from_genotypes(
        genotypes   = genotypes,
        model       = model,
        sample_name = sample_name
      )

      # Attach metadata columns
      if (nrow(mixture_df) > 0) {
        mixture_df$donor_ids  <- paste(combo, collapse=";")
        mixture_df$noc        <- noc
        mixture_df$ratios     <- paste(round(ratios, 4), collapse=";")
        mixture_df$sample_id  <- sample_name
        all_rows[[length(all_rows) + 1]] <- mixture_df
      }
    }, error = function(e) {
      cat(sprintf("    ERROR mix %d: %s\n", mix_i, conditionMessage(e)))
    })
  }
  n_simulated <- n_simulated + n_mix
}

# ── Combine & write output ───────────────────────────────────────────────────
if (length(all_rows) == 0) {
  stop("No mixtures simulated — check genotype data and spec")
}

out_df <- bind_rows(all_rows)
cat(sprintf("\nTotal rows: %d  from %d samples\n", nrow(out_df), n_simulated))
cat("Columns:", paste(names(out_df), collapse=", "), "\n")

write.csv(out_df, out_path, row.names = FALSE)
cat(sprintf("Saved -> %s\n", out_path))
