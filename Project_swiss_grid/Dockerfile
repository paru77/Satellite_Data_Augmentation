FROM continuumio/anaconda3

WORKDIR /app

COPY . /app

COPY  test /app/test

RUN conda env create -f enviroment.yml

EXPOSE 5000

# Command to run the Flask application
CMD ["conda", "run", "-n", "sentinel", "python", "app.py"]
