FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

# Copy the conda lockfile into the container
COPY conda-linux-64.lock /tmp/conda-linux-64.lock

# Install packages from lockfile, clean up environment and fix permissions
RUN mamba update --quiet --file /tmp/conda-linux-64.lock && \ 
    # FOLLOWING LINE CAUSES BUILD ERROR for Apple Silicon chips: mamba clean --all -y -f && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"