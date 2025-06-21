from pydantic import BaseModel
from typing import List

class CommentsRequest(BaseModel):
    comments: List[str]
