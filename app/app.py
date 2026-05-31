import os
import joblib
from flask import Flask, render_template, request, jsonify
from preprocessing import prepare_input
import asyncio
import httpx
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
bundle     = joblib.load(MODEL_PATH)
pipeline   = bundle['pipeline']
threshold  = bundle['best_threshold']

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

async def keep_alive():
    async with httpx.AsyncClient() as http:
        while True:
            await asyncio.sleep(600)
            try:
                await http.get("https://alexa-project.onrender.com/", timeout=5)
                logger.info("✅ Keep-alive ping отправлен")
            except Exception as e:
                logger.warning(f"⚠️ Keep-alive ошибка: {e}")

if __name__ == '__main__':
    import threading

    def run_keep_alive():
        asyncio.run(keep_alive())

    threading.Thread(target=run_keep_alive, daemon=True).start()

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)