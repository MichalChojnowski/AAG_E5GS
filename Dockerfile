#####################################################################################
# main
###############################################################################

FROM continuumio/miniconda3:4.8.2 AS main

RUN apt-get -y update && \
    apt-get -y install build-essential
RUN conda update -n base -c defaults conda

WORKDIR /opt/app

# chown changes owner from root owner (1000) to the first user inside the env (100)
COPY --chown=1000:100 requirements.txt .
RUN conda install --force-reinstall -y -q --name base -c conda-forge --file requirements.txt

# Install CARTOframes
RUN pip3 install cartoframes==1.0.4

COPY src .
COPY data data

EXPOSE 8888 8050

CMD python app.py

###############################################################################
# test
###############################################################################

FROM main AS test

COPY tests tests
RUN py.test tests
