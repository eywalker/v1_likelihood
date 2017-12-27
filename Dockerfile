FROM ninai/netgard
    
ADD . /src/v1_likelihood
RUN pip install -e /src/v1_likelihood

WORKDIR /notebooks


