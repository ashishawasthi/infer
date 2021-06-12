from flask import Flask
from flask import request, Response

from Model import PredictionError
from inference import get_prediction
import json
import logging
import os

logger = logging.getLogger('app')

app = Flask(__name__)
default_model_id = 'iris_svm_v1'  # TODO Load from application config database, remove hard-coding here
usage_guide = "Usage: /infer?model_id=registered_model_id&" \
              "model_inputs=[[comma_separated_inputs],[comma_separated_inputs]] "


@app.route('/')
def index():
    return usage_guide, 200


@app.route('/infer')
def infer():
    # Parse Request
    req_data = request.get_json()
    logger.debug(f'req_data: {req_data}')
    model_id = request.args.get('model_id')
    if model_id is None:
        model_id = default_model_id
    model_inputs_param = request.args.get('model_inputs')
    if model_inputs_param is None:
        logger.error(f'No model_inputs found, returning 400')
        return f'No model_inputs found. {usage_guide}', 400
    try:
        model_inputs = json.loads(model_inputs_param)
    except json.decoder.JSONDecodeError as e:
        logger.error(f'Non JSON model_inputs: {model_inputs_param}, returning 400. \n{str(e)}')
        return f'Non JSON model_inputs: {model_inputs_param}. {usage_guide}', 400
    logger.debug(f'model_inputs: {model_inputs}')

    # Predict
    try:
        prediction = get_prediction(model_id, model_inputs)
    except PredictionError as e:
        logger.error(f'model_inputs: {model_inputs_param}, returning 400. \n{str(e)}')
        return str(e), 400
    return Response(json.dumps(prediction.tolist()), mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8080)))
