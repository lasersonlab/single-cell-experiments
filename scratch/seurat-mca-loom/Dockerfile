FROM rocker/tidyverse

RUN apt-get install -y libhdf5-dev

RUN Rscript -e "devtools::install_github(repo = 'hhoeflin/hdf5r')"
RUN Rscript -e "devtools::install_github(repo = 'mojaveazure/loomR', ref = 'develop')"
RUN Rscript -e "install.packages('onlinePCA')" # needed o/w Seurat install fails
RUN Rscript -e "devtools::install_github(repo = 'satijalab/seurat', ref = 'loom')"
