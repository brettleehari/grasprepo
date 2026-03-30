"""Pydantic models for request validation and response shaping."""

import re
from pydantic import BaseModel, field_validator


class AnalyzeRequest(BaseModel):
    repo_url: str

    @field_validator("repo_url")
    @classmethod
    def validate_github_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        pattern = r"^https://github\.com/[\w.\-]+/[\w.\-]+$"
        if not re.match(pattern, v):
            raise ValueError(
                "Please provide a valid GitHub repository URL "
                "(e.g. https://github.com/user/repo)"
            )
        return v

    @property
    def repo_name(self) -> str:
        return self.repo_url.rstrip("/").split("/")[-1]

    @property
    def owner(self) -> str:
        return self.repo_url.rstrip("/").split("/")[-2]


class DownloadRequest(BaseModel):
    email: str | None = None
    linkedin: str | None = None
    repo_url: str
    repo_name: str
    ranked_files: list[dict] = []
    churn_files: list[dict] = []
    hotspots: list[dict] = []

    @field_validator("email", "linkedin", mode="before")
    @classmethod
    def at_least_one_contact(cls, v, info):
        return v

    def model_post_init(self, __context):
        if not self.email and not self.linkedin:
            raise ValueError("Please provide either an email address or a LinkedIn profile URL.")
