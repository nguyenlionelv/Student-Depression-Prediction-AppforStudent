all: install train test run deploy

install:
	pip install -r requirements.txt

train:
	python src/pipelines/pipeline.py

test:
# 	pytest tests/ -v
	python -m pytest tests/ -v

run:
	echo "window.RUNTIME_API_BASE = '$(API_BASE)';" > static/env.js
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

deploy:
	echo "window.RUNTIME_API_BASE = '$(API_BASE)';" > static/env.js
	python -m webbrowser "http://localhost:8000"
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

render_build:
	echo "window.RUNTIME_API_BASE = 'https://student-depression-predictation.onrender.com';" > static/env.js
	pip install -r requirements.txt

render_start:
	uvicorn api.main:app --host 0.0.0.0 --port 10000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
