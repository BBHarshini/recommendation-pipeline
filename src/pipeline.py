"""The ordered list of pipeline stages, shared by the Prefect flow and the plain Python runner."""

from src.evaluation import evaluate
from src.features import build as features
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.validation import validate

STAGES = [
    ("ingestion", ingest.run),
    ("validation", validate.run),
    ("preparation", prepare.run),
    ("transformation", transform.run),
    ("features", features.run),
    ("training", train.run),
    ("evaluation", evaluate.run),
]
