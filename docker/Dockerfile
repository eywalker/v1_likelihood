# Intermediate Container for private github repositories
FROM eywalker/attorch:latest as intermediate

ARG ssh_prv_key
ARG ssh_pub_key

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
        ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
        chmod 600 /root/.ssh/id_rsa && \
            chmod 600 /root/.ssh/id_rsa.pub

# Clone Repositories
WORKDIR /src
RUN git clone git@github.com:eywalker/v1_likelihood.git


# Final Container (without priavte SSH keys) with installed python packages

FROM eywalker/attorch:latest
COPY --from=intermediate /src/v1_likelihood /src/v1_likelihood
RUN pip install -e /src/v1_likelihood
RUN pip install h5py

WORKDIR /src/v1_likelihood/scripts
ENTRYPOINT ["python3"]
