from Model import Model, PredictionError
import logging

logger = logging.getLogger('infer')

# TODO Load from your model registry system, remove hard-coding here
models = {
    'iris_svm_v1': Model(model_id='iris_svm_v1', model_lib='sklearn', model_type='svm',
                         usage_example="/infer?model_id=iris_svm_v1&model_inputs=[[1,2,3,4],[4,3,2,1]] "),
    'ann_v1': Model(model_id='ann_v1', model_lib='tensorflow', model_type='ann', path='models/ann_v1.h5',
                        usage_example='/infer?model_id=ann_v1&model_inputs=[[1,2,3,4],[4,3,2,1]]')
    }


def get_prediction(model_id, X):
    if model_id not in models:
        raise PredictionError(f'Requested model_id {model_id} is not registered')
    req_model = models[model_id]
    try:
        model_object = req_model.get_model_object()
        prediction = model_object.predict(X)
    except PredictionError as pe:
        raise pe
    except Exception as e:
        raise PredictionError(f'Prediction failed with error: {str(e)}. Example usage: {req_model.usage_example}')
    logger.debug(f'predicted: {prediction}')
    return prediction


def register_model(model_id, model_lib, model_type, usage_example):
    if model_id not in models:
        models[model_id] = Model(model_id, model_lib=model_lib, model_type=model_type, usage_example=usage_example)
