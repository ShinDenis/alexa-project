import os
import joblib
from flask import Flask, render_template, request, jsonify
from preprocessing import prepare_input

# ── Загрузка модели ───────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
bundle     = joblib.load(MODEL_PATH)
pipeline   = bundle['pipeline']
threshold  = bundle['best_threshold']

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

VARIATIONS = ['Echo', 'Echo Dot', 'Echo Show', 'Echo Spot', 'Echo Plus', 'Fire TV Stick']

@app.route('/')
def index():
    return render_template('index.html', variations=VARIATIONS)


@app.route('/predict', methods=['POST'])
def predict():
    data       = request.get_json()
    review     = data.get('review', '').strip()
    variation  = data.get('variation', 'Echo')

    if not review:
        return jsonify({'error': 'Введите текст отзыва'}), 400

    input_df   = prepare_input(review, variation)
    proba      = pipeline.predict_proba(input_df)[0]
    proba_pos  = float(proba[1])
    proba_neg  = float(1 - proba[1])
    is_negative = proba_pos < threshold

    return jsonify({
        'label':     'Negative' if is_negative else 'Positive',
        'proba_pos': round(proba_pos * 100, 1),
        'proba_neg': round(proba_neg * 100, 1),
        'threshold': threshold
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
