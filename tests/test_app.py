import pytest
from app import app as flask_app
import json
import os.path
from sklearn import svm
from sklearn import datasets
import joblib

model_id = 'iris_svm_v1'


@pytest.fixture
def app():
    yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def model():
    filepath = 'models/' + model_id + '.pkl'
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
    json_response = json.loads(res.get_data(as_text=True))
    assert type(json_response) is list
    assert len(json_response) == 1


def test_infer_two_inferences(client, model):
    res = client.get(f'/infer?model_id={model}&model_inputs=[[1,2,3,4],[1,1,1,1]]')
    assert res.status_code == 200
    json_response = json.loads(res.get_data(as_text=True))
    assert type(json_response) is list
    assert len(json_response) == 2


def test_infer_wrong_input_count(client, model):
    res = client.get(f'/infer?model_id={model}&model_inputs=[[1,2,3]]')
    assert res.status_code == 400
    text_response = res.get_data(as_text=True)
    assert len(text_response) > 0


def test_infer_wrong_input_format(client, model):
    res = client.get(f'/infer?model_id={model}&model_inputs=[1,2,3,4]')
    assert res.status_code == 400
    text_response = res.get_data(as_text=True)
    assert len(text_response) > 0


def test_infer_missing_inputs(client):
    res = client.get('/infer')
    assert res.status_code == 400
    text_response = res.get_data(as_text=True)
    assert len(text_response) > 0
