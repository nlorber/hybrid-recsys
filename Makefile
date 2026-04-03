.PHONY: setup index test lint typecheck serve demo docker

setup:
	poetry install
	poetry run python -m spacy download fr_core_news_sm
	poetry run python -m spacy download en_core_web_sm
	poetry run python -m spacy download de_core_news_sm
	poetry run python -m nltk.downloader stopwords

index:
	poetry run python scripts/generate_catalog.py && poetry run hybrid-recsys index

test:
	poetry run pytest --tb=short

lint:
	poetry run ruff check src/ tests/

typecheck:
	poetry run mypy --strict src/

serve:
	poetry run hybrid-recsys serve

demo:
	poetry run hybrid-recsys demo "true crime podcast" --lang en --size 3

docker:
	docker compose up --build
