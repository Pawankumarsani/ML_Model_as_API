from fastapi import FastAPI
from fastapi.params import Body

app =FastAPI()

@app.get("/")
def root():
    return "RESTAPI using FASTAPI"

@app.post("/posts")
def posts(payload: dict = Body(   )):
    print(payload)
    return {"New_Post": f"Title: {payload['title']}, Content: {payload['content']}"}