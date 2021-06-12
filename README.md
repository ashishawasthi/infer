# Model Inference API example
Self-contained minimalistic ML model inference service example. 
The application demonstrates a wrapper API over models to serve the predictions in batches. 
It includes e2e tests for requests and prediction responses.

Application expects the provided model object to have an sklearn-style predict function.

# Local Development Environment
## Setup
```cmd
pip install -r requirements.txt
```
### Test
```cmd
python -m pytest
coverage run -m pytest
coverage report -m
```
### Run
```cmd
flask run
```
### Try
<flask_url>/infer?model_id=iris_svm_v1&model_inputs=[[1,2,3,4],[1,1,1,1]]

# Deloy Docker Image
```cmd
docker build
```

# Deloy on Google Cloud Run
```cmd
gcloud builds submit --tag gcr.io/<project-id>/infer
gcloud run deploy --image gcr.io/<project-id>/infer
```
