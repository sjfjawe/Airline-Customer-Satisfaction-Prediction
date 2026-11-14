from flask import Flask, request, jsonify, render_template_string
import pickle
from predict import load_model, predict

app = Flask(__name__)

model, dv, scaler = load_model()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Airline Satisfaction Predictor</title>
    <style>
        * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        .card { background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .section { margin-bottom: 24px; }
        .section-title { font-weight: 600; color: #555; margin-bottom: 12px; border-bottom: 2px solid #007bff; padding-bottom: 4px; }
        .form-row { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px; }
        .form-group { flex: 1; min-width: 150px; }
        label { display: block; font-size: 13px; color: #666; margin-bottom: 4px; }
        select, input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }
        select:focus, input:focus { outline: none; border-color: #007bff; }
        .rating-group { flex: 1; min-width: 120px; }
        button { width: 100%; padding: 14px; background: #007bff; color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; margin-top: 16px; }
        button:hover { background: #0056b3; }
        .result { margin-top: 24px; padding: 20px; border-radius: 8px; text-align: center; display: none; }
        .result.satisfied { background: #d4edda; color: #155724; }
        .result.dissatisfied { background: #f8d7da; color: #721c24; }
        .result h2 { margin: 0 0 8px 0; font-size: 24px; }
        .result p { margin: 0; font-size: 18px; }
        .probability { font-size: 48px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>✈️ Airline Satisfaction Predictor</h1>
    <div class="card">
        <form id="predictionForm">
            <div class="section">
                <div class="section-title">Passenger Info</div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Gender</label>
                        <select name="Gender">
                            <option value="Female">Female</option>
                            <option value="Male">Male</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Age</label>
                        <input type="number" name="Age" value="35" min="1" max="100">
                    </div>
                    <div class="form-group">
                        <label>Customer Type</label>
                        <select name="Customer_Type">
                            <option value="Loyal Customer">Loyal Customer</option>
                            <option value="disloyal Customer">Disloyal Customer</option>
                        </select>
                    </div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">Flight Info</div>
                <div class="form-row">
                    <div class="form-group">
                        <label>Type of Travel</label>
                        <select name="Type_of_Travel">
                            <option value="Business travel">Business Travel</option>
                            <option value="Personal Travel">Personal Travel</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Class</label>
                        <select name="Class">
                            <option value="Business">Business</option>
                            <option value="Eco Plus">Eco Plus</option>
                            <option value="Eco">Eco</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Flight Delayed?</label>
                        <select name="has_delay">
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">Service Ratings (0-5, 0 = Not Applicable)</div>
                <div class="form-row">
                    <div class="rating-group">
                        <label>Inflight Wifi</label>
                        <select name="Inflight_wifi_service">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                    <div class="rating-group">
                        <label>Departure/Arrival Time</label>
                        <select name="Departure/Arrival_time_convenient">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                    <div class="rating-group">
                        <label>Online Booking</label>
                        <select name="Ease_of_Online_booking">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                    <div class="rating-group">
                        <label>Gate Location</label>
                        <select name="Gate_location">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="rating-group">
                        <label>Food & Drink</label>
                        <select name="Food_and_drink">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                    <div class="rating-group">
                        <label>Online Boarding</label>
                        <select name="Online_boarding">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                    <div class="rating-group">
                        <label>Seat Comfort</label>
                        <select name="Seat_comfort">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                    <div class="rating-group">
                        <label>Entertainment</label>
                        <select name="Inflight_entertainment">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="rating-group">
                        <label>On-board Service</label>
                        <select name="On-board_service">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                    <div class="rating-group">
                        <label>Leg Room</label>
                        <select name="Leg_room_service">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                    <div class="rating-group">
                        <label>Baggage Handling</label>
                        <select name="Baggage_handling">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                    <div class="rating-group">
                        <label>Check-in Service</label>
                        <select name="Checkin_service">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="rating-group">
                        <label>Inflight Service</label>
                        <select name="Inflight_service">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                    <div class="rating-group">
                        <label>Cleanliness</label>
                        <select name="Cleanliness">
                            <option value="0">0</option><option value="1">1</option><option value="2">2</option>
                            <option value="3" selected>3</option><option value="4">4</option><option value="5">5</option>
                        </select>
                    </div>
                </div>
            </div>

            <button type="submit">Predict Satisfaction</button>
        </form>

        <div id="result" class="result">
            <h2 id="resultLabel"></h2>
            <div class="probability"><span id="resultProb"></span>%</div>
            <p>probability of satisfaction</p>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = isNaN(value) ? value : Number(value);
            });

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();

            const resultDiv = document.getElementById('result');
            const isSatisfied = result.label[0] === 'satisfied';
            resultDiv.className = 'result ' + (isSatisfied ? 'satisfied' : 'dissatisfied');
            resultDiv.style.display = 'block';
            document.getElementById('resultLabel').textContent = isSatisfied ? '😊 Satisfied' : '😐 Neutral / Dissatisfied';
            document.getElementById('resultProb').textContent = (result.probability[0] * 100).toFixed(1);
        });
    </script>
</body>
</html>
'''

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    result = predict(data, model, dv, scaler)
    return jsonify(result)

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
