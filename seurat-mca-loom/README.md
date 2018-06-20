Build a docker container

```bash
docker build -t seurat-mca-loom seurat-mca-loom
```

Start using it. (Note you'll need to increase your docker memory limit to at least 12GB.)

```bash
docker run -d -v ~/Downloads:/Downloads --name seurat-mca-loom seurat-mca-loom
docker exec -it seurat-mca-loom R
```

From https://satijalab.org/seurat/mca_loom.html

```R
library(loomR)
library(Seurat)
mca.matrix <- readRDS(file = "/Downloads/MCA/MCA_merged_mat.rds")
mca.metadata <- read.csv("/Downloads/MCA/MCA_All-batch-removed-assignments.csv",
    row.names = 1)

# Only keep annotated cells
cells.use <- which(x = colnames(x = mca.matrix) %in% rownames(x = mca.metadata))
mca.matrix <- mca.matrix[, cells.use]
mca.metadata <- mca.metadata[colnames(x = mca.matrix), ]
# Create the loom file
mca <- create(filename = "/Downloads/mca.loom", data = mca.matrix, display.progress = TRUE,
    calc.numi = TRUE)
# Leaves us with 242k cells
mca
```

The resulting loom file is around 400MB.

Clean up docker resources

```bash
docker stop seurat-mca-loom
docker rm seurat-mca-loom
```
