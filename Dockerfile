# use the miniforge base, make sure you specify a verion
FROM condaforge/miniforge3:25.9.1-0

# install make so we can run make clean in the container
RUN apt-get update && apt-get install -y make

# copy the lockfile into the container
COPY conda-lock.yml conda-lock.yml

# setup conda-lock
RUN conda install -n base -c conda-forge conda-lock -y

# install packages from lockfile into dockerlock environment
RUN conda-lock install -n dockerlock conda-lock.yml

# make dockerlock the default environment
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate dockerlock" >> ~/.bashrc

# set the default shell to use bash with login to pick up bashrc
# this ensures that we are starting from an activated dockerlock environment
SHELL ["/bin/bash", "-l", "-c"]

# sets the default working directory
# this is also specified in the compose file
WORKDIR /workspace

# run JupyterLab on container start
# uses the jupyterlab from the install environment
CMD ["bash"]