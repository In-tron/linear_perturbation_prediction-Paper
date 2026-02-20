library(tidyverse)

test_train <- rjson::fromJSON(file = "/tmp/working_dir/results/seed_1_adamson_split") %>%
  enframe(name = "train", value = "perturbation") %>%
  unnest(perturbation)

load_results <- function(path){
  preds <- rjson::fromJSON(file = file.path(path, "all_predictions.json"))
  names <- rjson::fromJSON(file = file.path(path, "gene_names.json"))
  tibble(perturbation = names(preds), prediction = unname(preds), gene = list(names)) %>%
    unnest(c(prediction, gene)) %>%
    mutate(perturbation = map_chr(str_split(perturbation, pattern = "[+_]", n = 2), \(x) {
      tmp <- if(all(x == "ctrl" | x == "")) "ctrl" else if(length(x) == 2) x else c(x, "ctrl")
      paste0(tmp, collapse = "+")
    }))
}

ground_truth <- load_results("/tmp/working_dir/results/ground_truth_results/") %>%
  rename(truth = prediction)
preds <- bind_rows(linear = load_results("/tmp/working_dir/results/linear_results/"),
                   # gears = load_results("/tmp/working_dir/results/gears_results/"), 
                   # scGPT = load_results("/tmp/working_dir/results/sgpt_results/"), 
                   .id = "method")

highest_expr_genes <- ground_truth %>%
  filter(perturbation == "ctrl") %>%
  slice_max(truth, n = 1000) %>%
  select(gene, baseline = truth)

res <- inner_join(preds, ground_truth, by = c("gene", "perturbation")) %>%
  inner_join(highest_expr_genes, by = "gene") %>%
  summarize(dist = sqrt(sum((truth - prediction)^2)), 
            pearson_delta = cor(prediction - baseline, truth - baseline), 
            .by = c("perturbation", "method")) 

res %>%
  inner_join(test_train, by = "perturbation") %>%
  filter( train != "train") %>%
  ggplot(aes(x = method, y = dist)) +
    ggbeeswarm::geom_quasirandom()