import pytest
from app import app as flask_app
import json
import os.path
from sklearn import svm
from sklearn import datasets
import joblib

from inference import register_model

model_id = 'iris_svm_v1'


@pytest.fixture
def app():
    yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def model():
    filepath = f'models/{model_id}.pkl'
    if not os.path.isfile(filepath):
        svm_clf = svm.SVC()
        X, y = datasets.load_iris(return_X_y=True)
        svm_clf.fit(X, y)
        joblib.dump(svm_clf, filepath, compress=1)
    return model_id


def test_index(client):
    res = client.get('/')
    assert res.status_code == 200
    assert 'infer' in res.get_data(as_text=True)


def test_infer_one_inference(client, model):
    res = client.get(f'/infer?model_id={model}&model_inputs=[[1,2,3,4]]')
    assert res.status_code == 200
    assert res.mimetype == 'application/json'
    json_response = json.loads(res.get_data(as_text=True))
    assert type(json_response) is list
    assert len(json_response) == 1


def test_infer_two_inferences(client, model):
    res = client.get(f'/infer?model_id={model}&model_inputs=[[1,2,3,4],[1,1,1,1]]')
    assert res.status_code == 200
    assert res.mimetype == 'application/json'
    json_response = json.loads(res.get_data(as_text=True))
    assert type(json_response) is list
    assert len(json_response) == 2


def test_infer_default_model_id(client):
    res = client.get(f'/infer?model_inputs=[[1,2,3,4],[1,1,1,1]]')
    assert res.status_code == 200
    assert res.mimetype == 'application/json'
    json_response = json.loads(res.get_data(as_text=True))
    assert type(json_response) is list
    assert len(json_response) == 2


def test_infer_non_registered_model_id(client, model):
    res = client.get(f'/infer?model_id={model}_non_registered&model_inputs=[[1,2,3,4],[1,1,1,1]]')
    assert res.status_code == 400
    text_response = res.get_data(as_text=True)
    assert len(text_response) > 0
    assert 'not registered' in text_response


def test_infer_wrong_input_count(client, model):
    res = client.get(f'/infer?model_id={model}&model_inputs=[[1,2,3]]')
    assert res.status_code == 400
    text_response = res.get_data(as_text=True)
    assert len(text_response) > 0
    assert 'model_inputs' in text_response


def test_infer_non_json_input(client, model):
    res = client.get(f'/infer?model_id={model}&model_inputs=not-a-json')
    assert res.status_code == 400
    text_response = res.get_data(as_text=True)
    assert len(text_response) > 0
    assert 'model_inputs' in text_response


def test_infer_wrong_input_format(client, model):
    res = client.get(f'/infer?model_id={model}&model_inputs=[1,2,3,4]')
    assert res.status_code == 400
    text_response = res.get_data(as_text=True)
    assert len(text_response) > 0
    assert 'model_inputs' in text_response


def test_infer_missing_inputs(client):
    res = client.get('/infer')
    assert res.status_code == 400
    text_response = res.get_data(as_text=True)
    assert len(text_response) > 0
    assert 'model_inputs' in text_response


#@pytest.mark.skip(reason="cannot be tested in parallel with other tests as file is created by others")
def test_infer_missing_model(client):
    new_model_id='iris_svm_v2'
    register_model(model_id=new_model_id, model_lib='sklearn', model_type='svm',
                   usage_example="/infer?model_id=iris_svm_v1&model_inputs=[[1,2,3,4],[4,3,2,1]] ")
    res = client.get(f'/infer?model_id={new_model_id}&model_inputs=[[1,2,3,4],[1,1,1,1]]')
    assert res.status_code == 400
    text_response = res.get_data(as_text=True)
    assert len(text_response) > 0
    assert 'path' in text_response
