"""Download the two outside models at the exact revisions the pipeline was built with.

They land in the Hugging Face cache (~/.cache/huggingface/hub/...), which is where src/bibi_reviews/paths.py looks.
The project's own models (post-trained encoder, sentiment classifier) come with the repository via Git LFS.
Usage: python scripts/setup_models.py
"""
from huggingface_hub import snapshot_download

MODELS = [
    ('nlpai-lab/KURE-v1', '8b418a58414668e75532ed045c22d9ca018ae2b2'),            # sentence embeddings (grouping)
    ('beomi/KcELECTRA-base-v2022', '568fe985a10c013c8f605e3c80a199413bbe9d00'),   # base of the post-training (3_kcelectra)
]

if __name__ == '__main__':
    for repo, revision in MODELS:
        print(repo, '->', snapshot_download(repo, revision=revision))
