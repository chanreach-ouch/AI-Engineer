run-docker-compose:
	uv sync
	docker compose up --build -d

clean-notebook-outputs:
	jupyter nbconvert -- clear-output -- inplace notebooks/*/ *.ipynb

run-evals-retriever:
	uv sync
	PYTHONPATH=${PWD}/app/api:${PWD}/app/api/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m app.api.evals.eval_retriever
