# Build context is the repo root (see template.yaml DockerContext: .)
FROM public.ecr.aws/lambda/python:3.12

COPY aws/classifier/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# train.py + predict.py provide clean() and classify() — reused, not reimplemented
COPY train.py predict.py aws/classifier/app.py ${LAMBDA_TASK_ROOT}/

CMD ["app.lambda_handler"]
