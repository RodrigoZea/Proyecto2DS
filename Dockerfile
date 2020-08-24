FROM jupyter/tensorflow-notebook

WORKDIR /home/jovyan/work

COPY ./requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt