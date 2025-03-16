from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from carpark_qa_system import CarparkQASystem
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("carpark_api")

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Enable CORS for all routes

# Initialize QA system (you might want to do this lazily on first request)
qa_system = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/api/query', methods=['POST'])
def process_query():
    global qa_system

    # Lazy initialization of QA system
    if qa_system is None:
        logger.info("Initializing Carpark QA System...")
        try:
            qa_system = CarparkQASystem()
            logger.info("QA System initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize QA System: {e}")
            return jsonify({
                "status": "error",
                "message": "System initialization failed"
            }), 500

    # Get query from request
    data = request.json
    if not data or 'query' not in data:
        return jsonify({
            "status": "error",
            "message": "Missing query parameter"
        }), 400

    query = data['query'].strip()
    if not query:
        return jsonify({
            "status": "error",
            "message": "Query cannot be empty"
        }), 400

    # Process query
    try:
        logger.info(f"Processing query: {query}")
        result = qa_system.answer_query(query)

        response = {
            "status": "success",
            "query": query,
            "answer": result["answer"],
            "carparks": [
                {
                    "region": cp.get("region", "N/A"),
                    "address": cp.get("address", "N/A"),
                    "area": cp.get("area", "N/A"),
                    "carpark_name": cp.get("carpark name", "N/A"),
                    "similarity": cp.get("similarity_score", 0)
                }
                for cp in result.get("carparks", [])
            ],
            "processing_time": result["processing_time"]
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error processing query: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
