# app/routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from .services import cluster_comments

router = APIRouter()

class CommentsRequest(BaseModel):
    comments: list[str]
    num_clusters: int = 10

@router.post("/cluster")
def cluster_endpoint(request: CommentsRequest):
    result = cluster_comments(request.comments, request.num_clusters)
    return {"questions": result}
